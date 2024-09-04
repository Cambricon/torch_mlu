#include <algorithm>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_dcn_forward_internal(
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const std::vector<int64_t>& output_sizes,
    int* padding,
    int* stride,
    int* dilation,
    int64_t deformable_group,
    int64_t conv_group,
    int64_t im2col_step,
    int bitwidth,
    bool use_mask) {
  const cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto output = at::empty(output_sizes, input.options(), memory_format);

  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto offset_impl = getMluTensorImpl(offset);
  auto offset_desc = getTensorDesc(offset_impl, layout);
  auto offset_ptr = mlu_data_ptr(offset_impl);

  auto weight_impl = getMluTensorImpl(weight);
  auto weight_desc = getTensorDesc(weight_impl, layout);
  auto weight_ptr = mlu_data_ptr(weight_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto output_ptr = mlu_data_ptr(output_impl);

  CnnlDCNDescriptor dcn_desc;
  dcn_desc.set(
      input.dim(),
      padding,
      stride,
      dilation,
      deformable_group,
      conv_group,
      im2col_step,
      CNNL_DTYPE_FLOAT);

  // prepare mask desc
  void* mask_ptr = nullptr;
  tensorDescPtr_t mask_desc;
  if (use_mask) {
    auto mask_impl = getMluTensorImpl(mask);
    mask_desc = getTensorDesc(mask_impl, layout);
    mask_ptr = mlu_data_ptr(mask_impl);
  }
  // prepare bias desc
  void* bias_ptr = nullptr;
  tensorDescPtr_t bias_desc;
  // bias Tensor size need be same with output Tensor channel size.
  if (bias.defined() && bias.dim() == 1 && bias.size(0) == output_sizes[1]) {
    TORCH_CHECK(
        bias.dim() == 1,
        "Currently only support 1-dim bias in dcn_forward_internal "
        "when bias.dim() != 0, but got ",
        bias.dim(),
        " dim.");
    auto bias_impl = getMluTensorImpl(bias);
    bias_desc = getTensorDesc(bias_impl, CNNL_LAYOUT_ARRAY);
    bias_ptr = mlu_data_ptr(bias_impl);
  }
  // get current handle
  auto handle = getCurrentHandle();

  size_t workspace_size = 0;

  // prepare workspace
  TORCH_CNNL_CHECK(cnnlGetDCNForwardWorkspaceSize(
      /* handle         */ handle,
      /* dcn_desc       */ dcn_desc.desc(),
      /* input_desc     */ input_desc.get(),
      /* offset_desc    */ offset_desc.get(),
      /* mask_desc      */ mask_desc.get(),
      /* weight_desc    */ weight_desc.get(),
      /* bias_desc      */ bias_desc.get(),
      /* output_desc    */ output_desc.get(),
      /* workspace_size */ &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlDCNForward(
      /* handle         */ handle,
      /* dcn_desc       */ dcn_desc.desc(),
      /* input_desc     */ input_desc.get(),
      /* input_ptr      */ input_ptr,
      /* offset_desc    */ offset_desc.get(),
      /* offset_ptr     */ offset_ptr,
      /* mask_desc      */ mask_desc.get(),
      /* mask_ptr       */ mask_ptr,
      /* weight_desc    */ weight_desc.get(),
      /* weight_ptr     */ weight_ptr,
      /* bias_desc      */ bias_desc.get(),
      /* bias_ptr       */ bias_ptr,
      /* workspace_ptr  */ workspace_ptr.get(),
      /* workspace_size */ workspace_size,
      /* output_desc    */ output_desc.get(),
      /* output_ptr     */ output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
