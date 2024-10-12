#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_masked_softmax_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const int axis,
    cnnlMaskedSoftmaxOp_t mode) {
  // set descriptor config
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  CnnlTensorDescriptor desc_mask;
  desc_input.set(input, CNNL_LAYOUT_ARRAY);
  desc_output.set(output, CNNL_LAYOUT_ARRAY);
  desc_mask.set(mask, CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto mask_impl = getMluTensorImpl(mask);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto mask_ptr = mask_impl->mlu_data_ptr();
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlMaskedSoftmax(
      handle,
      mode,
      axis,
      1.0,
      desc_input.desc(),
      input_ptr,
      desc_mask.desc(),
      mask_ptr,
      desc_output.desc(),
      output_ptr));

  return output;
}

} // namespace ops
} // namespace torch_mlu
