#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_as_strided_backward_internal(
    at::Tensor& grad_x,
    const at::Tensor& grad_y,
    at::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {
  auto input_impl = getMluTensorImpl(grad_y);
  auto descInput = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(grad_x);
  auto descOutput = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto handle = getCurrentHandle();

  c10::SmallVector<uint32_t, 8> strides;
  for (auto s : stride) {
    strides.push_back(static_cast<uint32_t>(s));
  }

  uint32_t offset = 0;
  if (storage_offset.has_value()) {
    offset = static_cast<uint32_t>(storage_offset.value());
  }

  uint32_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetAsStridedBackwardWorkspaceSize(
      handle, descOutput.get(), &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  TORCH_CNNL_CHECK(cnnlAsStridedBackward(
      handle,
      descInput.get(),
      input_ptr,
      descOutput.get(),
      output_ptr,
      strides.data(),
      offset,
      workspace_ptr.get(),
      space_size));

  return grad_x;
}

} // namespace ops
} // namespace torch_mlu
