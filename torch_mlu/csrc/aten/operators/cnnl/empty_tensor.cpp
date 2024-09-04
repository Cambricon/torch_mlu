#include <torch/csrc/utils/device_lazy_init.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "framework/core/mlu_guard.h"
#include "framework/hooks/MLUHooks.h"

namespace torch_mlu {
namespace ops {

// CAUTION : If you plan to call cnnl_empty/cnnl_empty_strided directly,
// BE SURE to handle the device guard by yourself.

// We deliberately skip lazy init and device guard in
// cnnl_empty/cnnl_empty_strided because on MLU they are always called through
// dispatch which handles lazy init and dispatch already. By skipping them a few
// hundred wasted CPU cycles are saved.

at::Tensor cnnl_empty(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided);

  const auto dtype = c10::dtype_or_default(dtype_opt);
  // at::globalContext().lazyInitPrivateUse1();
  // const auto device = c10::device_or_default(device_opt);
  // TORCH_INTERNAL_ASSERT(device.is_privateuseone());
  // const torch_mlu::mlu::MLUGuard device_guard(device);
  auto* allocator = torch_mlu::MLUCachingAllocator::get();
  constexpr c10::DispatchKeySet mlu_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(
      size, allocator, mlu_dks, dtype, memory_format_opt);
}

at::Tensor cnnl_empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_CHECK(
      !pin_memory_opt.has_value() || !*pin_memory_opt,
      "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  // at::globalContext().lazyInitPrivateUse1();
  // const auto device = c10::device_or_default(device_opt);
  // TORCH_INTERNAL_ASSERT(device.is_privateuseone());
  // const torch_mlu::mlu::MLUGuard device_guard(device);
  auto* allocator = torch_mlu::MLUCachingAllocator::get();
  constexpr c10::DispatchKeySet mlu_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, mlu_dks, dtype);
}

} // namespace ops
} // namespace torch_mlu
