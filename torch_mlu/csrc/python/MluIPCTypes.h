#pragma once
#include <c10/core/Allocator.h>
#include <c10/util/Logging.h>
#include <torch/csrc/Export.h>
#include <cstddef>
#include "cnrt.h"
#include "framework/core/caching_allocator.h"
#include "aten/utils/exceptions.h"

namespace torch_mlu {

C10_EXPORT bool MluIPCCollect();

struct MluIPCReceivedData final {
  MluIPCReceivedData() = default;
  explicit MluIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct MluIPCSentData final {
  std::string handle_;
  uint64_t offset_;
  uint64_t* counter_ptr_; // Reference counter shared memory block
  at::DataPtr original_ptr_; // Original mem allocation
  cnrtNotifier_t event_; // Sync cnrtNotifierDestroy
  bool event_sync_required_;
  at::Device device_;

  MluIPCSentData(
      std::string handle,
      uint64_t offset,
      uint64_t* counter_ptr,
      at::Device device);
  ~MluIPCSentData();

  uint64_t counter_value();
  std::string handle() {
    return handle_;
  }
  uint64_t offset() {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

C10_EXPORT at::DataPtr GetNewRefCountedSentData(void* data, at::Device device);

inline constexpr int64_t MLU_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t MLU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;
inline constexpr int64_t MLU_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

// All to be deleted data blocks with non zero reference counter goes there
struct MluIPCSentDataLimbo final {
  ~MluIPCSentDataLimbo();
  bool collect();
  void add(std::unique_ptr<MluIPCSentData> shared_block);
  uint64_t size();

 private:
  std::vector<std::unique_ptr<MluIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct MluIPCRefCountersFile final {
  MluIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size),

        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  uint64_t* counter_ptr() {
    return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }

  bool have_offsets() {
    return next_offset_ < size_;
  }

  bool offsets_in_use() {
    return used_slots_;
  }

  uint64_t get_offset() {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset /* unused */) {
    used_slots_--;
  }

  std::string handle() {
    return handle_;
  }

 private:
  uint64_t next_offset_{0};
  uint64_t size_;
  uint64_t used_slots_{0};
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

} // namespace torch_mlu

namespace torch_mlu::MLUCachingAllocator {
class MluIPCCollectCallback : public FreeMemoryCallback {
 public:
  bool Execute() override {
    return torch_mlu::MluIPCCollect();
  }
};
} // namespace torch_mlu::MLUCachingAllocator
