#include <ATen/detail/CUDAHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>
#include "framework/core/caching_allocator.h"
#include "framework/core/device.h"
#include "framework/core/memory_allocator.h"
#include "framework/hooks/MLUHooks.h"
#include "framework/core/mlu_guard.h"

// No need to have this whole header, we can just put it all in
// the cpp file

namespace torch_mlu {

bool hasPrimaryContext(int64_t device_index) {
  TORCH_CHECK(
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

at::Device MLUHooksInterface::getDeviceFromPtr(void* ptr) const {
  cnrtPointerAttributes_t attr;
  TORCH_CNRT_CHECK(cnrtPointerGetAttributes(&attr, static_cast<void*>(ptr)));
  TORCH_CHECK(
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

void MLUHooksInterface::initPrivateUse1() const {
  maybe_set_mlu_module_loading("LAZY");
  const auto num_devices = device_count_ensure_non_zero();
  torch_mlu::MLUCachingAllocator::init(num_devices);
}

c10::Allocator* MLUHooksInterface::getPinnedMemoryAllocator() const {
  return torch_mlu::getMLUCachingHostAllocator();
}

bool MLUHooksInterface::hasPrimaryContext(DeviceIndex device_index) const {
  return torch_mlu::hasPrimaryContext(device_index);
}

void MLUHooksInterface::resizePrivateUse1Bytes(
    const c10::Storage& storage,
    size_t newsize) const {
  auto storage_impl = storage.unsafeGetStorageImpl();
  TORCH_CHECK(
      storage_impl->resizable(),
      "Trying to resize storage that is not resizable");
  auto allocator = storage_impl->allocator();
  TORCH_CHECK(
      allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage_impl->device();

  if (newsize == 0) {
    storage_impl->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage_impl->set_nbytes(0);
    return;
  }

  torch_mlu::mlu::MLUGuard guard(device.index());
  at::DataPtr data = allocator->allocate(newsize);
  if (storage_impl->data_ptr()) {
    at::globalContext().lazyInitPrivateUse1();
    TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
        data.get(),
        const_cast<void*>(storage_impl->data()),
        std::min(storage_impl->nbytes(), newsize),
        torch_mlu::getCurrentMLUStream(),
        cnrtMemcpyDevToDev));
  }

  // Destructively overwrite data_ptr
  storage_impl->set_data_ptr_noswap(std::move(data));
  storage_impl->set_nbytes(newsize);
}

TORCH_DECLARE_REGISTRY(
    PrivateUse1HooksRegistry,
    MLUHooksInterface,
    MLUHooksArgs);
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, MLUHooksInterface, MLUHooksArgs)

static at::PrivateUse1HooksInterface* get_private_hooks() {
  static at::PrivateUse1HooksInterface* privateuse1_hooks;
  static c10::once_flag once;
  c10::call_once(once, [] {
    privateuse1_hooks =
        PrivateUse1HooksRegistry()->Create("PrivateUse1Hooks", {}).release();
    if (!privateuse1_hooks) {
      privateuse1_hooks = new MLUHooksInterface();
    }
  });
  return privateuse1_hooks;
}

int register_hook() {
  at::RegisterPrivateUse1HooksInterface(get_private_hooks());
  return 0;
}

// global static variable to make sure privateuse1hook is registered when
// torch_mlu.so is loaded.
// [Note register_hook]
// TODO(*): When modifying the logic for `lazy_init`, it
// is necessary to reconsider the entire context here.
static int dummy = register_hook();

} // namespace torch_mlu
