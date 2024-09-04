#include <gtest/gtest.h>

#include <c10/core/impl/InlineDeviceGuard.h>
#include "framework/core/guard_impl.h"

namespace torch_mlu {
using InlineMluGuard = c10::impl::InlineDeviceGuard<mlu::MLUGuardImpl>;

TEST(MLUGuardImplTest, constructor) {
  if (device_count() < 3)
    return;
  for (c10::DeviceIndex i : {-1, 0, 1}) {
    c10::DeviceIndex init_i = 0;
    cnrtSetDevice(init_i);
    auto test_body = [&](InlineMluGuard& g) -> void {
      ASSERT_EQ(
          g.original_device(),
          at::Device(at::Device::Type::PrivateUse1, init_i));
      ASSERT_EQ(
          g.current_device(),
          at::Device(at::Device::Type::PrivateUse1, (i == -1 ? init_i : i)));
      ASSERT_EQ(current_device(), (i == -1 ? init_i : i));
      cnrtSetDevice(2);
    };
    {
      // Index constructor
      InlineMluGuard g(i);
      test_body(g);
    }
    ASSERT_EQ(current_device(), init_i);
    {
      // Device constructor
      InlineMluGuard g(at::Device(at::Device::Type::PrivateUse1, i));
      test_body(g);
    }
    ASSERT_EQ(current_device(), init_i);
  }
}

TEST(MLUGuardImplTest, ConstructorError) {
  try {
    InlineMluGuard g(c10::Device(c10::DeviceType::CPU, 0));
  } catch (c10::Error& e) {
    std::cout << e.msg() << std::endl;
  }
}

TEST(MLUGuardImplTest, ResetDevice) {
  if (device_count() < 3)
    return;
  c10::DeviceIndex init_i = 0;
  cnrtSetDevice(init_i);
  c10::DeviceIndex i = init_i + 1;
  InlineMluGuard g(i);
  c10::DeviceIndex i2 = init_i + 2;
  g.reset_device(at::Device(at::Device::Type::PrivateUse1, i2));
  ASSERT_EQ(
      g.original_device(), at::Device(at::Device::Type::PrivateUse1, init_i));
  ASSERT_EQ(g.current_device(), at::Device(at::Device::Type::PrivateUse1, i2));
  ASSERT_EQ(current_device(), i2);
  g.reset_device(at::Device(at::Device::Type::PrivateUse1, i2));
  ASSERT_EQ(
      g.original_device(), at::Device(at::Device::Type::PrivateUse1, init_i));
  ASSERT_EQ(g.current_device(), at::Device(at::Device::Type::PrivateUse1, i2));
  ASSERT_EQ(current_device(), i2);
}

TEST(MLUGuardImplTest, SetIndex) {
  if (device_count() < 3)
    return;
  c10::DeviceIndex init_i = 0;
  cnrtSetDevice(init_i);
  c10::DeviceIndex i = init_i + 1;
  InlineMluGuard g(i);
  c10::DeviceIndex i2 = init_i + 2;
  g.set_index(i2);
  ASSERT_EQ(
      g.original_device(), at::Device(at::Device::Type::PrivateUse1, init_i));
  ASSERT_EQ(g.current_device(), at::Device(at::Device::Type::PrivateUse1, i2));
  ASSERT_EQ(current_device(), i2);
  g.set_index(i2);
  ASSERT_EQ(
      g.original_device(), at::Device(at::Device::Type::PrivateUse1, init_i));
  ASSERT_EQ(g.current_device(), at::Device(at::Device::Type::PrivateUse1, i2));
  ASSERT_EQ(current_device(), i2);
}

} // namespace torch_mlu
