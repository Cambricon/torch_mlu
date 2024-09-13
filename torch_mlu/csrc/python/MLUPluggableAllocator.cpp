#include <mutex>
#include <unordered_map>
#include <utility>
#include "python/MLUPluggableAllocator.h"

namespace torch_mlu::MLUPluggableAllocator {

MLUPluggableAllocatorDeleterContext::MLUPluggableAllocatorDeleterContext(
    std::function<FreeFuncType> free_fn,
    void* data,
    size_t size,
    int device,
    cnrtQueue_t stream)
    : free_fn_(free_fn),
      data_(data),
      size_(size),
      device_(device),
      stream_(stream) {}

void MLUPluggableAllocatorDeleterContext::free() {
  free_fn_(data_, size_, device_, stream_);
  delete this;
}

int device_count = 0;

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata()
    : size(0), device_idx(-1), stream{} {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
    cnrtQueue_t stream)
    : size(size), device_idx(device_idx), stream(stream) {}

// This is a fast API to just register allocators
// based on function pointers (ie. external .so libraries)
// This avoids having to link against libtorch for C++ based custom allocators
// And also use this from python
MLUPluggableAllocator::MLUPluggableAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn)
    : alloc_fn_(std::move(alloc_fn)), free_fn_(std::move(free_fn)) {}

MLUPluggableAllocator::MLUPluggableAllocator(MLUPluggableAllocator& other)
    : alloc_fn_(other.alloc_fn_),
      free_fn_(other.free_fn_),
      init_fn_(other.init_fn_),
      reset_fn_(other.reset_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

void MLUPluggableAllocator::set_init_fn(std::function<void(int)> init_fn) {
  init_fn_ = std::move(init_fn);
}

void MLUPluggableAllocator::set_reset_fn(std::function<void()> reset_fn) {
  reset_fn_ = std::move(reset_fn);
}

void MLUPluggableAllocator::set_memory_fraction_fn(
    std::function<void(double, int)> memory_fraction_fn) {
  memory_fraction_fn_ = std::move(memory_fraction_fn);
}

void MLUPluggableAllocator::set_base_alloc_fn(
    std::function<void*(void*, size_t*)> base_alloc_fn) {
  base_alloc_fn_ = std::move(base_alloc_fn);
}

void MLUPluggableAllocator::set_record_stream_fn(
    std::function<void(void* ptr, cnrtQueue_t stream)> record_stream_fn) {
  record_stream_fn_ = std::move(record_stream_fn);
}

void MLUPluggableAllocator::set_begin_allocate_to_pool(
    std::function<void(
        int,
        torch_mlu::MLUCachingAllocator::MempoolId_t,
        std::function<bool(cnrtQueue_t)>)> capture_begin_fn) {
  begin_allocate_to_pool_fn_ = std::move(capture_begin_fn);
}

void MLUPluggableAllocator::set_end_allocate_to_pool_fn(
    std::function<void(int, torch_mlu::MLUCachingAllocator::MempoolId_t)>
        capture_about_to_end_fn) {
  end_allocate_to_pool_fn_ = std::move(capture_about_to_end_fn);
}

void MLUPluggableAllocator::set_release_pool(
    std::function<void(int, torch_mlu::MLUCachingAllocator::MempoolId_t)>
        capture_destroy_fn) {
  relase_pool_fn_ = std::move(capture_destroy_fn);
}

void* MLUPluggableAllocator::malloc(
    size_t size,
    c10::DeviceIndex device,
    cnrtQueue_t stream) {
  void* r = alloc_fn_(size, device, stream);
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream));
  }
  return r;
}

c10::DataPtr MLUPluggableAllocator::allocate(size_t size) const {
  c10::DeviceIndex device = -1;
  device = torch_mlu::current_device();
  cnrtQueue_t stream = torch_mlu::getCurrentMLUStream(device);
  void* r =
      const_cast<MLUPluggableAllocator*>(this)->malloc(size, device, stream);
  auto* ctx = new MLUPluggableAllocatorDeleterContext(
      free_fn_, r, size, device, stream);
  c10::DataPtr data_ptr = {
      r, ctx, raw_deleter(), c10::Device(c10::DeviceType::PrivateUse1, device)};
  return data_ptr;
}

c10::DeleterFnPtr MLUPluggableAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* MLUPluggableAllocator::raw_alloc(size_t nbytes) {
  c10::DeviceIndex device = -1;
  device = torch_mlu::current_device();
  cnrtQueue_t stream = torch_mlu::getCurrentMLUStream(device);
  return malloc(nbytes, device, stream);
}

void* MLUPluggableAllocator::raw_alloc_with_stream(
    size_t nbytes,
    cnrtQueue_t stream) {
  c10::DeviceIndex device = -1;
  device = torch_mlu::current_device();
  return malloc(nbytes, device, stream);
}

void MLUPluggableAllocator::raw_delete(void* ptr) {
  cnrtQueue_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    TORCH_CHECK(
        allocation_metadata_.count(ptr),
        "Trying to free a pointer not allocated here");
    _AllocationMetadata& metadata = allocation_metadata_[ptr];
    size = metadata.size;
    device_idx = metadata.device_idx;
    stream = metadata.stream;
    allocation_metadata_.erase(ptr);
  }
  free_fn_(ptr, size, device_idx, stream);
}

void MLUPluggableAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool MLUPluggableAllocator::initialized() {
  return initialized_;
}

void MLUPluggableAllocator::setMemoryFraction(
    double fraction,
    c10::DeviceIndex device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

void MLUPluggableAllocator::emptyCache() {
  if (reset_fn_) {
    return reset_fn_();
  }
}

void MLUPluggableAllocator::cacheInfo(
    c10::DeviceIndex device,
    size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* MLUPluggableAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void MLUPluggableAllocator::recordStream(
    const c10::DataPtr& ptr,
    streamType stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

torch_mlu::MLUCachingAllocator::DeviceStats MLUPluggableAllocator::
    getDeviceStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void MLUPluggableAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void MLUPluggableAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

torch_mlu::MLUCachingAllocator::SnapshotInfo MLUPluggableAllocator::snapshot() {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support snapshot. "
      "If you need it, please file an issue describing your use case.");
}

torch_mlu::MLUCachingAllocator::ShareableHandle MLUPluggableAllocator::
    shareIpcHandle(void* ptr) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support shareIPcHandle. "
      "If you need it, please file an issue describing your use case.");
}

std::shared_ptr<void> MLUPluggableAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// MLUGraph interactions
void MLUPluggableAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id,
    std::function<bool(cnrtQueue_t)> filter) {
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void MLUPluggableAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id) {
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void MLUPluggableAllocator::releasePool(
    c10::DeviceIndex device,
    torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id) {
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void MLUPluggableAllocator::recordHistory(
    bool enabled,
    torch_mlu::MLUCachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    torch_mlu::MLUCachingAllocator::RecordContext when) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void MLUPluggableAllocator::attachOutOfMemoryObserver(
    torch_mlu::MLUCachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void MLUPluggableAllocator::attachAllocatorTraceTracker(
    torch_mlu::MLUCachingAllocator::AllocatorTraceTracker tracker) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not support attachAllocatorTraceTracker. "
      "attachAllocatorTraceTracker is only used inside Pytorch.");
}

std::shared_ptr<torch_mlu::MLUCachingAllocator::AllocatorState>
MLUPluggableAllocator::getCheckpointState(
    c10::DeviceIndex device,
    torch_mlu::MLUCachingAllocator::MempoolId_t id) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

torch_mlu::MLUCachingAllocator::CheckpointDelta MLUPluggableAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<torch_mlu::MLUCachingAllocator::AllocatorState> pps) {
  TORCH_CHECK(
      false,
      "MLUPluggableAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void MLUPluggableAllocator::enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  torch_mlu::mlu::MLUGuard device_guard(dev);
  unsigned int can_access = 0;
  cnrtRet_t err = cnrtGetPeerAccessibility(&can_access, dev_to_access, 0);
  if (err == cnrtSuccess) {
    // ignore and clear the error if access was already enabled
    (void)cnrtGetLastError();
  } else {
    TORCH_CNRT_CHECK(err);
  }
}

cnrtRet_t MLUPluggableAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cnrtQueue_t stream,
    bool p2p_enabled) {
  return cnrtMemcpyAsync_V2(
      dst, const_cast<void*>(src), count, stream, cnrtMemcpyDevToDev);
}

std::string MLUPluggableAllocator::name() {
  return "pluggable";
}

std::shared_ptr<torch_mlu::MLUCachingAllocator::MLUAllocator>
    current_custom_allocator;

std::shared_ptr<torch_mlu::MLUCachingAllocator::MLUAllocator>
getCurrentAllocator() {
  return current_custom_allocator;
}

std::shared_ptr<torch_mlu::MLUCachingAllocator::MLUAllocator>
createCustomAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn) {
  std::shared_ptr<MLUPluggableAllocator> allocator(
      new MLUPluggableAllocator(std::move(alloc_fn), std::move(free_fn)));
  allocator->init(device_count);
  return allocator;
}

void changeCurrentAllocator(
    const std::shared_ptr<torch_mlu::MLUCachingAllocator::MLUAllocator>&
        allocator) {
  TORCH_CHECK(
      !torch_mlu::MLUCachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  torch_mlu::MLUCachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ctx) {
  reinterpret_cast<MLUPluggableAllocatorDeleterContext*>(ctx)->free();
}

} // namespace torch_mlu::MLUPluggableAllocator
