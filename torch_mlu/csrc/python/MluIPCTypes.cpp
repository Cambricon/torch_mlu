#include <ATen/MapAllocator.h>
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include "framework/core/mlu_guard.h"
#include "python/MluIPCTypes.h"

namespace torch_mlu {

void warnProducerTerminatedBeforeSharedTensorsReleased() {
  static bool warned = false;
  if (!warned) {
    LOG(WARNING)
        << "Producer process has been terminated before all shared MLU tensors released. See Note [Sharing MLU tensors]";
    warned = true;
  }
}

struct MluIPCGlobalEntities {
  // This class is used as a singleton (see mlu_ipc_global_entities)
  // This variable is used to track its lifetime to avoid accessing it
  // after it was destroyed which would lead to segmentation faults
  // Note that a trvial type is used which doesn't suffer from construction
  // and destruction order issues
  static bool alive;

  std::mutex ref_counters_mutex_;
  std::atomic<int64_t> sync_events_used_{0};
  std::map<std::string, std::shared_ptr<MluIPCRefCountersFile>>
      ref_counters_files_;
  std::shared_ptr<MluIPCRefCountersFile> next_available_ref_counters_file_;
  MluIPCSentDataLimbo MluIPCSentDataLimbo_;
  MluIPCGlobalEntities() {
    alive = true;
  }
  ~MluIPCGlobalEntities() {
    MluIPCSentDataLimbo_.collect();
    safe_clean_current_file();
    if (next_available_ref_counters_file_) {
      warnProducerTerminatedBeforeSharedTensorsReleased();
    }
    alive = false;
  }
  void safe_clean_current_file() {
    std::lock_guard<std::mutex> lock(ref_counters_mutex_);
    if (next_available_ref_counters_file_ &&
        next_available_ref_counters_file_->offsets_in_use() == 0) {
      ref_counters_files_.erase(next_available_ref_counters_file_->handle());
      next_available_ref_counters_file_.reset();
    }
  }
};

bool MluIPCGlobalEntities::alive = false;
MluIPCGlobalEntities mlu_ipc_global_entities;

MluIPCSentDataLimbo::~MluIPCSentDataLimbo() {
  collect();
  if (size() > 0) {
    warnProducerTerminatedBeforeSharedTensorsReleased();
  }
}

bool MluIPCSentDataLimbo::collect() {
  bool freed_memory = false;
  std::vector<std::unique_ptr<MluIPCSentData>> reset_blocks;
  { // Begin critical section to modify shared blocks
    std::lock_guard<std::mutex> lock(limbo_mutex_);
    std::vector<std::unique_ptr<MluIPCSentData>> kept_blocks;
    for (auto& sd : shared_blocks_) {
      if (sd->counter_value() > 0) {
        kept_blocks.push_back(std::move(sd));
      } else {
        freed_memory = true;
        reset_blocks.push_back(std::move(sd));
      }
    }
    shared_blocks_ = std::move(kept_blocks);
  }
  // Need to reset blocks out of the critical section here, otherwise it
  // deadlocks.
  for (auto& sd : reset_blocks) {
    sd.reset();
  }
  return freed_memory;
}

void MluIPCSentDataLimbo::add(std::unique_ptr<MluIPCSentData> shared_block) {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  static bool warned = false;
  if (shared_blocks_.size() > MLU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO && !warned) {
    LOG(WARNING)
        << "Producer process tried to deallocate over "
        << MLU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO
        << " memory blocks referred by consumer processes. Deallocation might be significantly slowed down. "
        << "We assume it will never going to be the case.";
    warned = true;
  }
  shared_blocks_.push_back(std::move(shared_block));
}

uint64_t MluIPCSentDataLimbo::size() {
  std::lock_guard<std::mutex> lock(limbo_mutex_);
  return shared_blocks_.size();
}

void MluIPCSentDataDelete(void* ptr) {
  std::unique_ptr<MluIPCSentData> sent_data(static_cast<MluIPCSentData*>(ptr));
  if (!MluIPCGlobalEntities::alive) {
    return;
  }
  if (sent_data->counter_value() > 0) {
    mlu_ipc_global_entities.MluIPCSentDataLimbo_.add(std::move(sent_data));
  }
  mlu_ipc_global_entities.MluIPCSentDataLimbo_.collect();
}

void ReturnRefCounter(const std::string& handle, uint64_t offset /* unused */) {
  if (!MluIPCGlobalEntities::alive) {
    return;
  }
  std::lock_guard<std::mutex> lock(mlu_ipc_global_entities.ref_counters_mutex_);
  auto& map = mlu_ipc_global_entities.ref_counters_files_;
  auto it = map.find(handle);
  if (it != map.end()) {
    it->second->return_offset(offset);
    if (it->second->offsets_in_use() == 0 && !it->second->have_offsets()) {
      map.erase(handle);
    }
  }
}

MluIPCSentData::MluIPCSentData(
    std::string handle,
    uint64_t offset,
    uint64_t* counter_ptr,
    at::Device device)
    : handle_(std::move(handle)),
      offset_(offset),
      counter_ptr_(counter_ptr),
      original_ptr_(),
      device_(device) {
  if (mlu_ipc_global_entities.sync_events_used_.load() <
      MLU_IPC_MAXIMUM_EVENTS_TO_USE) {
    mlu_ipc_global_entities.sync_events_used_++;
    TORCH_CNRT_CHECK(cnrtNotifierCreateWithFlags(
        &event_,
        CNRT_NOTIFIER_DISABLE_TIMING_ALL | CNRT_NOTIFIER_INTERPROCESS));
    TORCH_CNRT_CHECK(cnrtPlaceNotifier(
        event_, torch_mlu::getCurrentMLUStream(device.index())));
    event_sync_required_ = true;
  } else {
    auto stream = torch_mlu::getCurrentMLUStream(device.index());
    TORCH_CNRT_CHECK(cnrtQueueSync(stream));
    event_ = nullptr;
    event_sync_required_ = false;
  }
}

MluIPCSentData::~MluIPCSentData() {
  ReturnRefCounter(handle_, offset_);
  try {
    if (event_sync_required_) {
      torch_mlu::mlu::MLUGuard device_guard(device_.index());
      TORCH_CNRT_CHECK(cnrtNotifierDestroy(event_));
      if (!MluIPCGlobalEntities::alive) {
        return;
      }
      mlu_ipc_global_entities.sync_events_used_--;
    }
  } catch (...) { /* No throw */
  }
}

uint64_t MluIPCSentData::counter_value() {
  return *counter_ptr_;
}

at::DataPtr GetNewRefCountedSentData(void* data, at::Device device) {
  {
    std::lock_guard<std::mutex> lock(
        mlu_ipc_global_entities.ref_counters_mutex_);
    if (!mlu_ipc_global_entities.next_available_ref_counters_file_) {
      std::string ref_counter_handle = at::NewProcessWideShmHandle();

      int flags =
          at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE;
      at::DataPtr sptr = at::RefcountedMapAllocator::makeDataPtr(
          ref_counter_handle.c_str(),
          flags,
          sizeof(int64_t) * MLU_IPC_REF_COUNTER_FILE_SIZE,
          nullptr);
      auto rc = std::make_shared<MluIPCRefCountersFile>(
          ref_counter_handle, MLU_IPC_REF_COUNTER_FILE_SIZE, std::move(sptr));
      mlu_ipc_global_entities.ref_counters_files_[ref_counter_handle] = rc;
      mlu_ipc_global_entities.next_available_ref_counters_file_ = rc;
    }
  }
  mlu_ipc_global_entities.next_available_ref_counters_file_->set_counter(1);
  auto sent_data = new MluIPCSentData(
      mlu_ipc_global_entities.next_available_ref_counters_file_->handle(),
      mlu_ipc_global_entities.next_available_ref_counters_file_->get_offset(),
      mlu_ipc_global_entities.next_available_ref_counters_file_->counter_ptr(),
      device);

  mlu_ipc_global_entities.next_available_ref_counters_file_->rotate_offset();
  if (!mlu_ipc_global_entities.next_available_ref_counters_file_
           ->have_offsets()) {
    mlu_ipc_global_entities.next_available_ref_counters_file_.reset();
  }
  return at::DataPtr(data, sent_data, MluIPCSentDataDelete, device);
}

bool MluIPCCollect() {
  if (!MluIPCGlobalEntities::alive) {
    return true;
  }
  bool freed_memory = mlu_ipc_global_entities.MluIPCSentDataLimbo_.collect();
  if (mlu_ipc_global_entities.MluIPCSentDataLimbo_.size() == 0) {
    mlu_ipc_global_entities.safe_clean_current_file();
  }
  return freed_memory;
}

} // namespace torch_mlu

namespace torch_mlu::MLUCachingAllocator {
REGISTER_FREE_MEMORY_CALLBACK("mlu_ipc_collect", MluIPCCollectCallback);
}
