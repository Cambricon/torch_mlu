#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <thread>
#include "framework/core/caching_allocator.h"
#include "framework/core/device.h"
#include "framework/core/device_utils.h"
#include "framework/core/MLUStream.h"
#include "framework/core/stream_guard.h"
#include "c10/util/Optional.h"
#include "c10/util/Logging.h"

namespace torch_mlu {

TEST(StreamTest, getCurrentMLUStreamTest) {
  {
    auto stream = getCurrentMLUStream();
    auto default_stream = getDefaultMLUStream();
    TORCH_CHECK_EQ(stream, default_stream);
  }
  for (int i = 0; i < device_count(); ++i) {
    auto stream = getCurrentMLUStream(i);
    auto default_stream = getDefaultMLUStream(i);
    TORCH_CHECK_EQ(stream, default_stream);
  }
}

TEST(StreamTest, getCnrtQueueTest) {
  {
    auto stream = getCurMLUStream();
    auto default_stream = getDefaultMLUStream();
    TORCH_CHECK_EQ(stream, default_stream.stream());
    default_stream.synchronize();
  }
  for (int i = 0; i < device_count(); ++i) {
    auto stream = getCurMLUStream(i);
    auto default_stream = getDefaultMLUStream(i);
    TORCH_CHECK_EQ(stream, default_stream.stream());
  }
}

TEST(StreamTest, CopyAndMoveTest) {
  int32_t device = -1;
  cnrtQueue_t cnrt_stream;

  auto copyStream = getStreamFromPool();
  {
    auto stream = getStreamFromPool();
    device = stream.device_index();
    cnrt_stream = stream.stream();

    copyStream = stream;

    TORCH_CHECK_EQ(copyStream.device_index(), device);
    TORCH_CHECK_EQ(copyStream.stream(), cnrt_stream);
  }

  TORCH_CHECK_EQ(copyStream.device_index(), device);
  TORCH_CHECK_EQ(copyStream.stream(), cnrt_stream);

  auto moveStream = getStreamFromPool();
  {
    auto stream = getStreamFromPool();
    device = stream.device_index();
    cnrt_stream = stream.stream();

    moveStream = std::move(stream);

    TORCH_CHECK_EQ(moveStream.device_index(), device);
    TORCH_CHECK_EQ(moveStream.stream(), cnrt_stream);
  }
  TORCH_CHECK_EQ(moveStream.device_index(), device);
  TORCH_CHECK_EQ(moveStream.stream(), cnrt_stream);
}

TEST(StreamTest, GetAndSetTest) {
  auto myStream = getStreamFromPool();

  // sets and gets
  setCurrentMLUStream(myStream);
  auto curStream = getCurrentMLUStream();

  TORCH_CHECK_EQ(myStream, curStream);

  // Gets, sets, and gets the default stream
  auto defaultStream = getDefaultMLUStream();
  setCurrentMLUStream(defaultStream);
  curStream = getCurrentMLUStream();

  TORCH_CHECK_NE(defaultStream, myStream);
  TORCH_CHECK_EQ(curStream, defaultStream);
}

void thread_fun(at::optional<MLUStream>& cur_thread_stream, int device) {
  for (int i = 0; i < 50; i++) {
    auto new_stream = getStreamFromPool();
    setCurrentMLUStream(new_stream);
    cur_thread_stream = {getCurrentMLUStream()};
    TORCH_CHECK_EQ(*cur_thread_stream, new_stream);
  }
}

TEST(StreamTest, MultithreadGetAndSetTest) {
  at::optional<MLUStream> s0, s1;
  std::thread t0{thread_fun, std::ref(s0), 0};
  std::thread t1{thread_fun, std::ref(s1), 0};
  t0.join();
  t1.join();
  auto cur_stream = getCurrentMLUStream();
  auto default_stream = getDefaultMLUStream();
  EXPECT_EQ(cur_stream, default_stream);
  EXPECT_NE(cur_stream, *s0);
  EXPECT_NE(cur_stream, *s1);
}

TEST(StreamTest, StreamPoolTest) {
  std::vector<MLUStream> streams{};
  for (int i = 0; i < 200; ++i) {
    streams.emplace_back(getStreamFromPool());
  }

  std::unordered_set<cnrtQueue_t> stream_set{};
  bool hasDuplicates = false;
  for (auto i = decltype(streams.size()){0}; i < streams.size(); ++i) {
    auto mlu_stream = streams[i].stream();
    auto result_pair = stream_set.insert(mlu_stream);
    if (!result_pair.second)
      hasDuplicates = true;
  }
  EXPECT_TRUE(hasDuplicates);
}

TEST(StreamTest, MultiMLUTest) {
  if (device_count() < 2)
    return;

  auto s0 = getStreamFromPool(false, 0);
  auto s1 = getStreamFromPool(false, 1);
  setCurrentMLUStream(s0);

  TORCH_CHECK_EQ(s0, getCurrentMLUStream());
  setCurrentMLUStream(s1);
  setDevice(1);
  TORCH_CHECK_EQ(s1, getCurrentMLUStream());
}

TEST(StreamTest, StreamGuardTest) {
  auto original_stream = getCurrentMLUStream();
  auto stream = getStreamFromPool();
  torch_mlu::mlu::MLUStreamGuard guard(stream);
  TORCH_CHECK_EQ(stream, getCurrentMLUStream());
  TORCH_CHECK_NE(original_stream, getCurrentMLUStream());
  TORCH_CHECK_EQ(guard.current_stream(), getCurrentMLUStream());
  TORCH_CHECK_EQ(guard.original_stream(), original_stream);
  auto nstream = getStreamFromPool();
  guard.reset_stream(nstream);
  TORCH_CHECK_EQ(nstream, getCurrentMLUStream());
}

TEST(StreamTest, StreamQuery) {
  const size_t size = 100 * 1024 * 1024;
  auto stream = getCurrentMLUStream();
  ASSERT_TRUE(stream.query());
  torch_mlu::MLUCachingAllocator::init(device_count());
  auto allocator = torch_mlu::MLUCachingAllocator::get();
  auto src_ptr = allocator->allocate(size);
  auto dst_ptr = allocator->allocate(size);
  cnrtMemcpyAsync_V2(
      dst_ptr.get(), src_ptr.get(), size, stream.stream(), cnrtMemcpyDevToDev);
  ASSERT_FALSE(stream.query());
  stream.synchronize();
  ASSERT_TRUE(stream.query());
}

} // namespace torch_mlu
