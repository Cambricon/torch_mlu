#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_masked_softmax_dropout_backward_internal(
    at::Tensor& diff_x,
    const at::Tensor& softmax_out,
    const at::Tensor& diff_y,
    const at::Tensor& dropout_mask,
    const int axis,
    const float p) {
  // set descriptor config
  CnnlTensorDescriptor desc_diff_x;
  CnnlTensorDescriptor desc_diff_y;
  CnnlTensorDescriptor desc_softmax_out;
  CnnlTensorDescriptor desc_dropout_mask;
  desc_diff_x.set(diff_x, CNNL_LAYOUT_ARRAY);
  desc_diff_y.set(diff_y, CNNL_LAYOUT_ARRAY);
  desc_softmax_out.set(softmax_out, CNNL_LAYOUT_ARRAY);
  desc_dropout_mask.set(dropout_mask, CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto diff_x_impl = getMluTensorImpl(diff_x);
  auto diff_y_impl = getMluTensorImpl(diff_y);
  auto softmax_out_impl = getMluTensorImpl(softmax_out);
  auto dropout_mask_impl = getMluTensorImpl(dropout_mask);
  auto diff_x_ptr = diff_x_impl->mlu_data_ptr();
  auto diff_y_ptr = diff_y_impl->mlu_data_ptr();
  auto softmax_out_ptr = softmax_out_impl->mlu_data_ptr();
  auto dropout_mask_ptr = dropout_mask_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlMaskedScaleSoftmaxBackward(
      handle,
      axis,
      1.0,
      desc_softmax_out.desc(),
      softmax_out_ptr,
      desc_diff_y.desc(),
      diff_y_ptr,
      desc_dropout_mask.desc(),
      static_cast<const uint8_t*>(dropout_mask_ptr),
      p,
      desc_diff_x.desc(),
      diff_x_ptr));

  return diff_x;
}

} // namespace ops
} // namespace torch_mlu
