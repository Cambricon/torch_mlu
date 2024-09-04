#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include <ATen/CPUFunctions.h>
#include "c10/core/Storage.h"
namespace torch_mlu {
namespace ops {

bool cnnl_is_pinned(const at::Tensor& self, std::optional<at::Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !device.has_value() || device->is_privateuseone());
  auto data_ptr = self.data_ptr();
  return isPinnedPtr(data_ptr);
}

at::Tensor cnnl__pin_memory(
    const at::Tensor& self,
    std::optional<at::Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !device.has_value() || device->is_privateuseone());
  auto* allocator = getCachingHostAllocator();

  const size_t storage_size = at::detail::computeStorageNbytes(
      self.sizes(), self.strides(), self.dtype().itemsize());
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(), storage_size, allocator, false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace ops
} // namespace torch_mlu
