#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>
#include "framework/core/device.h"
#include "framework/hooks/MLUHooks.h"
#include "framework/core/caching_allocator.h"
#include "aten/operators/cnnl/internal/cnfft_plan_cache.h"
#include "aten/utils/utils.h"

// No need to have this whole header, we can just put it all in
// the cpp file

namespace torch_mlu {

bool hasPrimaryContext(int64_t device_index) {
  TORCH_MLU_CHECK(
      device_index >= 0 && device_index < device_count(),
      "hasPrimaryContext expects a valid device index, but got device_index=",
      device_index);
  unsigned int flag = 0;
  int state = 0;
  TORCH_CNDRV_CHECK(cnSharedContextGetState(
      static_cast<int64_t>(device_index), &flag, &state));
  // CN_CONTEXT_STATE_ACTIVE enum value is 1.
  return state == CN_CONTEXT_STATE_ACTIVE;
}

c10::optional<int64_t> getDeviceIndexWithSharedContext() {
  int16_t current_device_index = current_device();
  if (hasPrimaryContext(current_device_index))
    return current_device_index;
  int16_t num_devices = device_count();
  for (int16_t device_index = 0; device_index < num_devices; ++device_index) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index))
      return device_index;
  }
  return c10::nullopt;
}

at::Device MLUHooks::getDeviceFromPtr(void* ptr) const {
  cnrtPointerAttributes_t attr;
  TORCH_CNRT_CHECK(cnrtPointerGetAttributes(&attr, static_cast<void*>(ptr)));
  TORCH_MLU_CHECK(
      attr.type == cnrtMemTypeUnregistered,
      "Memeory is not registered in MLU side.");
  return {at::kPrivateUse1, static_cast<c10::DeviceIndex>(attr.device)};
}

// Sets the CN_MODULE_LOADING environment variable
// if it's not set by the user.
void maybe_set_mlu_module_loading(const std::string& def_value) {
  auto value = std::getenv("CN_MODULE_LOADING");
  if (!value) {
#ifdef _WIN32
    auto env_var = "CN_MODULE_LOADING=" + def_value;
    _putenv(env_var.c_str());
#else
    setenv("CN_MODULE_LOADING", def_value.c_str(), 1);
#endif
  }
}

void MLUHooks::initCUDA() const {
  maybe_set_mlu_module_loading("LAZY");
  const auto num_devices = device_count_ensure_non_zero();
  torch_mlu::MLUCachingAllocator::init(num_devices);
}

c10::Allocator* MLUHooks::getPinnedMemoryAllocator() const {
  return getMLUCachingHostAllocator();
}

bool MLUHooks::isPinnedPtr(const void* data) const {
  return torch_mlu::isPinnedPtr(data);
}

int64_t MLUHooks::cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const {
  return ops::detail::cnfft_get_plan_cache_max_size_impl(device_index);
}

void MLUHooks::cuFFTSetPlanCacheMaxSize(
    DeviceIndex device_index,
    int64_t max_size) const {
  ops::detail::cnfft_set_plan_cache_max_size_impl(device_index, max_size);
}

int64_t MLUHooks::cuFFTGetPlanCacheSize(DeviceIndex device_index) const {
  return ops::detail::cnfft_get_plan_cache_size_impl(device_index);
}

void MLUHooks::cuFFTClearPlanCache(DeviceIndex device_index) const {
  ops::detail::cnfft_clear_plan_cache_impl(device_index);
}

bool MLUHooks::hasPrimaryContext(DeviceIndex device_index) const {
  return torch_mlu::hasPrimaryContext(device_index);
}

// Sigh, the registry doesn't support namespaces :(
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

REGISTER_CUDA_HOOKS(MLUHooks);

} // namespace torch_mlu
