#pragma once

#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/Generator.h>
#include <c10/util/Optional.h>
#include "framework/core/device.h"
#include "framework/core/memory_allocator.h"

// No need to have this whole header, we can just put it all in
// the cpp file

namespace torch_mlu {

TORCH_MLU_API bool hasPrimaryContext(int64_t device_index);
c10::optional<int64_t> getDeviceIndexWithSharedContext();
void maybe_set_mlu_module_loading(const std::string &def_value);

// The real implementation of CUDAHooksInterface
struct MLUHooks : public at::CUDAHooksInterface {
  explicit MLUHooks(at::CUDAHooksArgs) {}
  void initCUDA() const override;
  bool isPinnedPtr(const void* data) const override;
  at::Device getDeviceFromPtr(void* ptr) const override;
  c10::Allocator* getPinnedMemoryAllocator() const override;
  int64_t cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const override;
  void cuFFTSetPlanCacheMaxSize(DeviceIndex device_index, int64_t max_size)
      const override;
  int64_t cuFFTGetPlanCacheSize(DeviceIndex device_index) const override;
  void cuFFTClearPlanCache(DeviceIndex device_index) const override;
  bool hasPrimaryContext(DeviceIndex device_index) const override;
};

} // namespace torch_mlu
