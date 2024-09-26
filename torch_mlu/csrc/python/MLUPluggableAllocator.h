#pragma once
#include <c10/core/Allocator.h>
#include "framework/core/MLUStream.h"
#include "framework/core/caching_allocator.h"
#include "framework/core/mlu_guard.h"
#include <mutex>

namespace torch_mlu::MLUPluggableAllocator {

using MallocFuncType = void*(size_t, int, cnrtQueue_t);
using FreeFuncType = void(void*, size_t, int, cnrtQueue_t);

// A MLUPluggableAllocatorDeleterContext object is used as the `ctx`
// argument for DataPtr. We need context because a user can use
// multiple allocators in the same PyTorch program, and
// the allocators can have different free functions, such as:
// free, cnrtFree etc.
struct MLUPluggableAllocatorDeleterContext {
  explicit MLUPluggableAllocatorDeleterContext(
      std::function<FreeFuncType> free_fn,
      void* data,
      size_t size,
      int device,
      cnrtQueue_t stream);

  void free();

 private:
  std::function<FreeFuncType> free_fn_;
  void* data_;
  size_t size_;
  int device_;
  cnrtQueue_t stream_;
};

using streamType = torch_mlu::MLUStream;

std::shared_ptr<torch_mlu::MLUCachingAllocator::MLUAllocator>
getCurrentAllocator();
std::shared_ptr<torch_mlu::MLUCachingAllocator::MLUAllocator>
createCustomAllocator(
    std::function<MallocFuncType> alloc_fn,
    std::function<FreeFuncType> free_fn);
void changeCurrentAllocator(
    const std::shared_ptr<torch_mlu::MLUCachingAllocator::MLUAllocator>&
        allocator);

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(
      size_t size,
      c10::DeviceIndex device_idx,
      cnrtQueue_t stream);
  size_t size;
  c10::DeviceIndex device_idx;
  cnrtQueue_t stream;
};

struct MLUPluggableAllocator
    : public torch_mlu::MLUCachingAllocator::MLUAllocator {
  MLUPluggableAllocator(
      std::function<MallocFuncType> alloc_fn,
      std::function<FreeFuncType> free_fn);

  MLUPluggableAllocator(MLUPluggableAllocator& other);
  MLUPluggableAllocator& operator=(MLUPluggableAllocator& other) = delete;

  void set_init_fn(std::function<void(int)> init_fn);

  void set_reset_fn(std::function<void()> reset_fn);

  void set_memory_fraction_fn(
      std::function<void(double, int)> memory_fraction_fn);

  void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);

  void set_record_stream_fn(
      std::function<void(void* ptr, cnrtQueue_t stream)> record_stream_fn);

  void set_begin_allocate_to_pool(
      std::function<void(
          int,
          torch_mlu::MLUCachingAllocator::MempoolId_t,
          std::function<bool(cnrtQueue_t)>)> capture_begin_fn);

  void set_end_allocate_to_pool_fn(
      std::function<void(int, torch_mlu::MLUCachingAllocator::MempoolId_t)>
          capture_about_to_end_fn);

  void set_release_pool(
      std::function<void(int, torch_mlu::MLUCachingAllocator::MempoolId_t)>
          capture_destroy_fn);

  void* malloc(size_t size, c10::DeviceIndex device, cnrtQueue_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, cnrtQueue_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  void emptyCache() override;
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;

  void recordStream(const c10::DataPtr&, streamType stream) override;

  torch_mlu::MLUCachingAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  torch_mlu::MLUCachingAllocator::SnapshotInfo snapshot() override;
  void beginAllocateToPool(
      c10::DeviceIndex device,
      torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id,
      std::function<bool(cnrtQueue_t)>) override;
  void endAllocateToPool(
      c10::DeviceIndex device,
      torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id) override;
  void releasePool(
      c10::DeviceIndex device,
      torch_mlu::MLUCachingAllocator::MempoolId_t mempool_id) override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  // TODO: MLUAllocator does not support shareIpcHandle.
  torch_mlu::MLUCachingAllocator::ShareableHandle shareIpcHandle(void*);
  void recordHistory(
      bool enabled,
      torch_mlu::MLUCachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      torch_mlu::MLUCachingAllocator::RecordContext when) override;
  // TODO: MLUAllocator does not support attachOutOfMemoryObserver.
  void attachOutOfMemoryObserver(
      torch_mlu::MLUCachingAllocator::OutOfMemoryObserver observer);
  void attachAllocatorTraceTracker(
      torch_mlu::MLUCachingAllocator::AllocatorTraceTracker tracker) override;
  std::shared_ptr<torch_mlu::MLUCachingAllocator::AllocatorState>
  getCheckpointState(
      c10::DeviceIndex device,
      torch_mlu::MLUCachingAllocator::MempoolId_t id) override;
  torch_mlu::MLUCachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<torch_mlu::MLUCachingAllocator::AllocatorState> pps)
      override;
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override;
  cnrtRet_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cnrtQueue_t stream,
      bool p2p_enabled) override;
  std::string name() override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

 protected:
  std::function<MallocFuncType> alloc_fn_;
  std::function<FreeFuncType> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, cnrtQueue_t stream)> record_stream_fn_;
  std::function<void(
      int,
      torch_mlu::MLUCachingAllocator::MempoolId_t,
      std::function<bool(cnrtQueue_t)>)>
      begin_allocate_to_pool_fn_;
  std::function<void(int, torch_mlu::MLUCachingAllocator::MempoolId_t)>
      end_allocate_to_pool_fn_;
  std::function<void(int, torch_mlu::MLUCachingAllocator::MempoolId_t)>
      relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace torch_mlu::MLUPluggableAllocator
