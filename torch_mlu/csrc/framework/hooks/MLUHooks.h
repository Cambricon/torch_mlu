#pragma once

#include <ATen/Generator.h>
#include <ATen/Context.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/util/Optional.h>
#include "framework/generator/generator_impl.h"

// No need to have this whole header, we can just put it all in
// the cpp file

namespace torch_mlu {

TORCH_MLU_API bool hasPrimaryContext(int64_t device_index);
std::optional<int64_t> getDeviceIndexWithSharedContext();
int register_hook();

struct MLUHooksArgs : public at::PrivateUse1HooksArgs {};

struct MLUHooksInterface : public at::PrivateUse1HooksInterface {
  ~MLUHooksInterface() override = default;
  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) override {
    static auto device_gen = torch_mlu::getDefaultMLUGenerator(device_index);
    return device_gen;
  }
  at::Device getDeviceFromPtr(void* data) const override;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  void initPrivateUse1() const override;

  c10::Allocator* getPinnedMemoryAllocator() const override;

  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize)
      const override;
};

} // namespace torch_mlu
