#include <gtest/gtest.h>

#include "framework/core/MLUStream.h"
#include "framework/core/device.h"
#include "framework/core/stream_guard.h"

namespace torch_mlu {

static c10::Device dev(c10::DeviceIndex index) {
  return c10::Device(at::kPrivateUse1, index);
}

using TestGuard = mlu::MLUStreamGuard;

TEST(MLUStreamGuard, Constructor) {
  if (device_count() < 2) {
    return;
  }
  auto stream_0 = getDefaultMLUStream(0);
  cnrtSetDevice(0);
  {
    auto stream_1 = getStreamFromPool(false, 1);
    TestGuard g(stream_1);
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
    ASSERT_EQ(g.original_stream(), stream_0);
    ASSERT_EQ(g.current_stream(), stream_1);
    ASSERT_EQ(getCurrentMLUStream(1), stream_1);
    ASSERT_EQ(getCurrentMLUStream(0), stream_0);
    ASSERT_EQ(dev(current_device()), dev(1));
  }
  ASSERT_EQ(getCurrentMLUStream(0), stream_0);
  ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  ASSERT_EQ(dev(current_device()), dev(0));
  ASSERT_FALSE(std::is_trivially_constructible<TestGuard>::value);
  ASSERT_FALSE(std::is_trivially_copy_constructible<TestGuard>::value);
  ASSERT_FALSE(std::is_trivially_move_constructible<TestGuard>::value);
}

TEST(MLUStreamGuard, ResetStreamSameSatmeDevice) {
  cnrtSetDevice(0);
  auto stream_0 = getDefaultMLUStream(0);
  {
    auto stream_1 = getStreamFromPool(false, 0);
    auto stream_2 = getStreamFromPool(false, 0);
    TestGuard g(stream_1);
    g.reset_stream(stream_2);
    ASSERT_EQ(current_device(), 0);
    ASSERT_EQ(getCurrentMLUStream(0), stream_2);
    ASSERT_EQ(g.original_stream(), stream_0);
    ASSERT_EQ(g.current_stream(), stream_2);
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(0));
  }
  ASSERT_EQ(current_device(), 0);
  ASSERT_EQ(getCurrentMLUStream(0), stream_0);
}

TEST(MLUStreamGuard, ResetStreamDifferentSameDevice) {
  if (device_count() < 2) {
    return;
  }
  cnrtSetDevice(0);
  auto stream_0 = getDefaultMLUStream(0);
  {
    auto stream_1 = getStreamFromPool(0, 1);
    auto stream_2 = getStreamFromPool(0, 1);
    TestGuard g(stream_1);
    g.reset_stream(stream_2);
    ASSERT_EQ(current_device(), 1);
    ASSERT_EQ(getCurrentMLUStream(1), stream_2);
    ASSERT_EQ(getCurrentMLUStream(0), stream_0);
    ASSERT_EQ(g.original_stream(), stream_0);
    ASSERT_EQ(g.current_stream(), stream_2);
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
  }
  ASSERT_EQ(current_device(), 0);
  ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  ASSERT_EQ(getCurrentMLUStream(0), stream_0);
}

TEST(MLUStreamGuard, ResetStreamDifferentDevice) {
  if (device_count() < 3) {
    return;
  }
  cnrtSetDevice(0);
  {
    auto stream_1 = getStreamFromPool(0, 1);
    auto stream_2 = getStreamFromPool(0, 2);
    TestGuard g(stream_1);
    g.reset_stream(stream_2);
    ASSERT_EQ(current_device(), 2);
    ASSERT_EQ(getCurrentMLUStream(2), stream_2);
    ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
    ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
    ASSERT_EQ(g.original_stream(), getDefaultMLUStream(0));
    ASSERT_EQ(g.current_stream(), stream_2);
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(2));
  }
  ASSERT_EQ(current_device(), 0);
  ASSERT_EQ(getCurrentMLUStream(2), getDefaultMLUStream(2));
  ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
}

using MultiTestGuard = mlu::MLUMultiStreamGuard;

TEST(MLUMultiStreamGuard, Constructor) {
  if (device_count() < 2) {
    return;
  }
  {
    std::vector<MLUStream> streams;
    MultiTestGuard g(streams);
    ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
    ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  }
  ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
  ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  {
    std::vector<MLUStream> streams = {getStreamFromPool(0, 0)};
    MultiTestGuard g(streams);
    ASSERT_EQ(getCurrentMLUStream(0), streams[0]);
    ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  }
  ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
  ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  {
    std::vector<MLUStream> streams = {getStreamFromPool(0, 1)};
    MultiTestGuard g(streams);
    ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
    ASSERT_EQ(getCurrentMLUStream(1), streams[0]);
  }
  ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
  ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  {
    std::vector<MLUStream> streams = {
        getStreamFromPool(0, 0), getStreamFromPool(0, 1)};
    MultiTestGuard g(streams);
    ASSERT_EQ(getCurrentMLUStream(0), streams[0]);
    ASSERT_EQ(getCurrentMLUStream(1), streams[1]);
  }
  ASSERT_EQ(getCurrentMLUStream(0), getDefaultMLUStream(0));
  ASSERT_EQ(getCurrentMLUStream(1), getDefaultMLUStream(1));
  ASSERT_FALSE(std::is_trivially_copy_constructible<MultiTestGuard>::value);
  ASSERT_FALSE(std::is_trivially_move_constructible<MultiTestGuard>::value);
}

} // namespace torch_mlu
