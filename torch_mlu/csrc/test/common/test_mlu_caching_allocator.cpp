#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <thread>

#include <ATen/Context.h>
#include "ATen/ATen.h"
#include "c10/util/Optional.h"
#include "framework/core/caching_allocator.h"
#include "framework/core/caching_allocator_config.h"
#include "framework/core/device.h"
#include "framework/core/device_utils.h"
#include "framework/core/MLUStream.h"
#include "python/MLUPluggableAllocator.h"

namespace torch_mlu {

const int iterations = 100;
const size_t size = 10 * 1024;
const size_t free_size = 100 * 1024 * 1024; // 100 Mibs
const size_t large_buffer_size = 36 * 1024 * 1024; // 36 Mibs

static int segmentAllocCalled = 0;
static int segmentFreeCalled = 0;

static void SegmentAllocTraceTracker(
    const MLUCachingAllocator::TraceEntry& te) {
  if (te.action_ == MLUCachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    segmentAllocCalled++;
  }
}

static void SegmentFreeTraceTracker(const MLUCachingAllocator::TraceEntry& te) {
  if (te.action_ == MLUCachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    segmentFreeCalled++;
  }
}

static void allocateLargeBuffer() {
  const auto _500mb = 500 * 1024 * 1024;
  auto* allocator = MLUCachingAllocator::get();
  auto buffer = allocator->allocate(_500mb);
}

TEST(AllocatorTraceTracker, TrackMallocFree) {
  const auto num_devices = device_count();
  torch_mlu::MLUCachingAllocator::init(num_devices);
  MLUCachingAllocator::attachAllocatorTraceTracker(&SegmentAllocTraceTracker);
  MLUCachingAllocator::attachAllocatorTraceTracker(&SegmentFreeTraceTracker);

  // Expect to trigger segment allocation for large buffer
  // and expect the buffer would be marked as inactive when return from
  // allocateLargeBuffer and be freed when calling emptyCache
  allocateLargeBuffer();
  ASSERT_EQ(segmentAllocCalled, 1);

  // Expect allocated buffer has been released back to allocator, thus empty
  // cache would trigger segment free
  MLUCachingAllocator::emptyCache();
  ASSERT_EQ(segmentFreeCalled, 1);
}

TEST(MLUCachingAllocatorTest, allocate) {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  int16_t device = current_device();
  for (int i = 0; i < iterations; ++i) {
    auto data_ptr = ca->allocate(size);
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, emptyCache) {
  for (int s = 0; s < iterations; ++s) {
    auto ca = torch_mlu::MLUCachingAllocator::get();
    int16_t device = current_device();
    for (int i = 0; i < iterations; ++i) {
      auto data_ptr = ca->allocate(size * size);
      TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
    }
    MLUCachingAllocator::emptyCache();
  }
}

void thread_func() {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  int16_t device = current_device();
  for (int i = 0; i < iterations; i++) {
    auto data_ptr = ca->allocate(size);
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, allocateMultiThread) {
  for (int i = 0; i < 100; ++i) {
    std::thread t{thread_func};
    t.join();
  }
}

TEST(MLUCachingAllocatorTest, allocateMultiDevice) {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  for (int d = 0; d < device_count(); ++d) {
    setDevice(d);
    int16_t device = current_device();
    for (int i = 0; i < iterations; ++i) {
      auto data_ptr = ca->allocate(size);
      TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
    }
  }
}

TEST(MLUCachingAllocatorTest, recordStream) {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  int16_t device = current_device();
  for (int i = 0; i < iterations; ++i) {
    auto data_ptr = ca->allocate(size);
    MLUCachingAllocator::recordStream(data_ptr, getStreamFromPool());
    TORCH_CNRT_CHECK(cnrtMemset(data_ptr.get(), 1, size));
  }
}

TEST(MLUCachingAllocatorTest, getAllocationSize) {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  size_t malloc_size = free - free_size;
  auto data_ptr0 = ca->allocate(malloc_size);
  size_t free_size0 = 0;
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free_size0, &total));
  auto data_ptr1 = ca->allocate(large_buffer_size);
  size_t free_size1 = 0;
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free_size1, &total));
  size_t diff = free_size0 - free_size1;
  TORCH_CHECK(diff == large_buffer_size, "diff not equal large_buffer_size!");
}

TEST(MLUCachingAllocatorTest, free_available_cached_chunks_case1) {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  // Empty reserved memory in other case
  MLUCachingAllocator::emptyCache();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  auto data_ptr1 = ca->allocate(free_size * 2); // chunk1 -> 200 MB
  auto data_ptr2 = ca->allocate(free_size * 3); // chunk2 -> 300 MB
  auto data_ptr3 = ca->allocate(free - free_size * 5.5); // chunk3

  // data_ptr is a instance of c10:DataPtr (a wrapper of unique_ptr)
  // this will call the `MLUCachingDeleter`, so the chunk will be cached.
  data_ptr1.clear(); // chunk1 cached
  data_ptr2.clear(); // chunk2 cached

  // try to allocate 300 MB, this will free available cached chunks which is
  // chunk2. since chunk1 is less than 300 MB.
  auto data_ptr4 = ca->allocate(free_size * 3);
  auto cached = free_size * 2; // chunk1 cached in chunk pool, total 200 MB.
  auto stats = MLUCachingAllocator::mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  auto freed = stats["reserved_bytes.all.freed"];
  auto used = stats["active_bytes.all.current"];

  ASSERT_GE(reserved - freed - used, cached);
}

TEST(MLUCachingAllocatorTest, free_available_cached_chunks_case2) {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  // Empty reserved memory in other case
  MLUCachingAllocator::emptyCache();
  size_t max_split_size = 200;
  std::string env = "max_split_size_mb:" + std::to_string(max_split_size);
  torch_mlu::MLUCachingAllocator::setAllocatorSettings(env);
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  auto data_ptr1 = ca->allocate(free_size); // chunk1 -> 100 MB
  auto data_ptr2 = ca->allocate(free_size * 2); // chunk2 -> 200 MB
  auto data_ptr3 = ca->allocate(free_size * 2); // chunk3 -> 200 MB
  auto data_ptr4 = ca->allocate(free - free_size * 5.5); // chunk4

  // data_ptr is a instance of c10:DataPtr (a wrapper of unique_ptr)
  // this will call the `MLUCachingDeleter`, so the chunk will be cached.
  data_ptr1.clear(); // chunk1 cached
  data_ptr2.clear(); // chunk2 cached
  data_ptr3.clear(); // chunk3 cached

  // try to allocate 250 MB, this will free oversized chunks until to fit the
  // size requested, so chunk2 and chunk3 will be freed since they are greater
  // than max_split_size (200 MB)
  auto data_ptr5 = ca->allocate(free_size * 2.5);
  auto cached = free_size; // chunk1 cached in chunk pool, total 100 MB.
  auto stats = MLUCachingAllocator::mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  auto freed = stats["reserved_bytes.all.freed"];
  auto used = stats["active_bytes.all.current"];

  ASSERT_GE(reserved - freed - used, cached);
  // set to default value, do not affect other test case.
  MLUCachingAllocator::setAllocatorSettings("");
}

TEST(MLUCachingAllocatorTest, roundup_power2_divisions) {
  size_t roundup_bypass_threshold_mb = 1280;
  size_t roundup_power2_divisions = 4;
  std::string env =
      "roundup_power2_divisions:" + std::to_string(roundup_power2_divisions);
  MLUCachingAllocator::setAllocatorSettings(env);
  int16_t device = current_device();
  auto ca = torch_mlu::MLUCachingAllocator::get();
  auto data_ptr0 = ca->allocate(
      free_size * 11); // 1100 MB -> 1280 MB [1024, 1280, 1536, 1792, 2048]
  auto stats = MLUCachingAllocator::mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  size_t gt = 1280 * 1024 * 1024; // roundup to 1280 MB.

  ASSERT_GE(reserved, gt);

  // Empty reserved memory in other case
  MLUCachingAllocator::emptyCache();
  auto data_ptr1 = ca->allocate(
      free_size * 15); // 1500 MB -> 1500 MB (1500 > threshold 1280)
  stats = MLUCachingAllocator::mlu_memory_stats(device);
  reserved = stats["reserved_bytes.all.allocated"];
  gt = 1500 * 1024 * 1024;

  ASSERT_GE(reserved, gt);
  // set to default value, do not affect other test case.
  MLUCachingAllocator::setAllocatorSettings("");
}

TEST(MLUCachingAllocatorTest, garbage_collection) {
  double garbage_collection_threshold = 0.8;
  std::string env = "garbage_collection_threshold:" +
      std::to_string(garbage_collection_threshold);
  MLUCachingAllocator::setAllocatorSettings(env);
  auto ca = torch_mlu::MLUCachingAllocator::get();
  // Empty reserved memory in other case
  MLUCachingAllocator::emptyCache();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  double fraction = 0.8;
  MLUCachingAllocator::setMemoryFraction(fraction, device);
  auto data_ptr1 = ca->allocate(free_size); // chunk1 -> 100 MB
  auto data_ptr2 = ca->allocate(free_size); // chunk2 -> 100 MB
  auto data_ptr3 = ca->allocate(free_size * 2); // chunk3 -> 200 MB
  auto data_ptr4 = ca->allocate(free_size * 2); // chunk4 -> 200 MB
  auto data_ptr5 = ca->allocate(free * 0.8 - free_size * 6); // chunk5

  data_ptr1.clear(); // chunk1 cached
  data_ptr3.clear(); // chunk3 cached

  // try to allocate 300 MB chunk, this will free chunk1 and chun3
  // so there are no chunks cached.
  auto data_ptr6 = ca->allocate(free_size * 3);
  auto stats = MLUCachingAllocator::mlu_memory_stats(device);
  auto reserved = stats["reserved_bytes.all.allocated"];
  auto freed = stats["reserved_bytes.all.freed"];
  auto used = stats["active_bytes.all.current"];

  ASSERT_GE(reserved - freed - used, 0);
  // set to default value, do not affect other test case.
  MLUCachingAllocator::setAllocatorSettings("");
}

TEST(MLUCachingAllocatorTest, set_memory_fraction) {
  auto ca = torch_mlu::MLUCachingAllocator::get();
  // Empty reserved memory in other case
  MLUCachingAllocator::emptyCache();
  int16_t device = current_device();
  size_t free = 0;
  size_t total = 0;
  // get free memory size(MiB)
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
  double fraction = 0.6;
  MLUCachingAllocator::setMemoryFraction(fraction, device);
  bool exception_flag = true;
  // case 1 allocate more than 0.6 * free size, throw exception
  try {
    auto data_ptr = ca->allocate(free * 0.8);
    exception_flag = false;
  } catch (c10::Error) {
    ASSERT_GE(1, 0); // True
  }
  if (!exception_flag) {
    ASSERT_GE(0, 1); // False
  }
  // case 2: allocate less than 0.6 * free size, success.
  try {
    auto data_ptr1 = ca->allocate(free * 0.5);
    exception_flag = false;
  } catch (c10::Error) {
    ASSERT_GE(0, 1); // False
  }
  if (!exception_flag) {
    ASSERT_GE(1, 0); // True
  }
}

static int called_dummy_free_0 = 0;
static int called_dummy_free_1 = 0;

void* dummy_alloc_0(size_t size, int device, void* stream) {
  return nullptr;
}
void dummy_free_0(void* data, size_t size, int device, void* stream) {
  called_dummy_free_0++;
}
void dummy_free_1(void* data, size_t size, int device, void* stream) {
  called_dummy_free_1++;
}

// Tests that data_ptrs have their respective deleters
// when mixing allocators
TEST(AllocatorTestMLU, test_pluggable_allocator_deleters) {
  // Create a tensor with dummy_allocator_0, where dummy_free_0 is the deleter
  auto dummy_allocator_0 =
      torch_mlu::MLUPluggableAllocator::createCustomAllocator(
          dummy_alloc_0, dummy_free_0);
  torch_mlu::MLUCachingAllocator::allocator.store(dummy_allocator_0.get());
  at::Tensor a = at::empty({0}, at::TensorOptions().device(at::kPrivateUse1));

  // Create a tensor with dummy_allocator_1, where dummy_free_1 is the deleter
  auto dummy_allocator_1 =
      torch_mlu::MLUPluggableAllocator::createCustomAllocator(
          dummy_alloc_0, dummy_free_1);
  torch_mlu::MLUCachingAllocator::allocator.store(dummy_allocator_1.get());
  at::Tensor b = at::empty({0}, at::TensorOptions().device(at::kPrivateUse1));

  // Manually use a's deleter
  auto* ctx = a.storage().data_ptr().get_context();
  a.storage().data_ptr().get_deleter()(ctx);
  a.storage().mutable_data_ptr().release_context();

  // a's deleter is dummy_free_0
  // dummy_free_0 should be called above, so called_dummy_free_0 should be 1
  ASSERT_TRUE(called_dummy_free_0 == 1);

  // Manually use b's deleter
  ctx = b.storage().data_ptr().get_context();
  b.storage().data_ptr().get_deleter()(ctx);
  b.storage().mutable_data_ptr().release_context();

  // b's deleter is dummy_free_1
  // dummy_free_1 should be called above, so called_dummy_free_1 should be 1
  ASSERT_TRUE(called_dummy_free_1 == 1);
}

} // namespace torch_mlu
