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
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto output = at::empty(output_sizes, input.options(), memory_format);
  auto input_impl = getMluTensorImpl(input);
  auto offset_impl = getMluTensorImpl(offset);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_impl = getMluTensorImpl(output);
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor offset_desc;
  CnnlTensorDescriptor mask_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor bias_desc;
  CnnlTensorDescriptor output_desc;
  CnnlDCNDescriptor dcn_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();
  // desc set function
  auto desc_set = [&](const at::Tensor& t,
                      CnnlTensorDescriptor& t_desc,
                      cnnlTensorLayout_t layout) {
    auto shape_vec =
        modify_dims_based_on_layout(t.sizes().vec(), memory_format);
    auto stride_vec = get_contiguous_strides(shape_vec);
    t_desc.set(t, shape_vec, stride_vec, layout);
  };
  // prepare desc
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  desc_set(input, input_desc, layout);
  desc_set(offset, offset_desc, layout);
  desc_set(weight, weight_desc, layout);
  desc_set(output, output_desc, layout);
  cnnlSetTensorDescriptorOnchipDataType(
      output_desc.desc(), getCnnlDataType(output.dtype()));
  dcn_desc.set(
      input.dim(),
      padding,
      stride,
      dilation,
      deformable_group,
      conv_group,
      im2col_step,
      CNNL_DTYPE_FLOAT);
  // set onchip dtype
  cnnlSetTensorDescriptorOnchipDataType(
      input_desc.desc(), getCnnlDataType(input.dtype()));
  cnnlSetTensorDescriptorOnchipDataType(
      offset_desc.desc(), getCnnlDataType(offset.dtype()));
  cnnlSetTensorDescriptorOnchipDataType(
      weight_desc.desc(), getCnnlDataType(weight.dtype()));
  // prepare mask desc
  void* mask_ptr = nullptr;
  if (use_mask) {
    auto mask_impl = getMluTensorImpl(mask);
    desc_set(mask, mask_desc, layout);
    mask_ptr = mask_impl->mlu_data_ptr();
  }
  // prepare bias desc
  void* bias_ptr = nullptr;
  // bias Tensor size need be same with output Tensor channel size.
  if (bias.defined() && bias.dim() == 1 && bias.size(0) == output_sizes[1]) {
    TORCH_MLU_CHECK(
        bias.dim() == 1,
        "Currently only support 1-dim bias in dcn_forward_internal "
        "when bias.dim() != 0, but got ",
        bias.dim(),
        " dim.");
    auto bias_impl = getMluTensorImpl(bias);
    // layout = CNNL_LAYOUT_ARRAY;
    bias_desc.set(bias, CNNL_LAYOUT_ARRAY);
    bias_ptr = bias_impl->mlu_data_ptr();
  }

  // prepare workspace
  TORCH_CNNL_CHECK(cnnlGetDCNForwardWorkspaceSize(
      /* handle         */ handle,
      /* dcn_desc       */ dcn_desc.desc(),
      /* input_desc     */ input_desc.desc(),
      /* offset_desc    */ offset_desc.desc(),
      /* mask_desc      */ mask_desc.desc(),
      /* weight_desc    */ weight_desc.desc(),
      /* bias_desc      */ bias_desc.desc(),
      /* output_desc    */ output_desc.desc(),
      /* workspace_size */ &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto offset_ptr = offset_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  TORCH_CNNL_CHECK(cnnlDCNForward(
      /* handle         */ handle,
      /* dcn_desc       */ dcn_desc.desc(),
      /* input_desc     */ input_desc.desc(),
      /* input_ptr      */ input_ptr,
      /* offset_desc    */ offset_desc.desc(),
      /* offset_ptr     */ offset_ptr,
      /* mask_desc      */ mask_desc.desc(),
      /* mask_ptr       */ mask_ptr,
      /* weight_desc    */ weight_desc.desc(),
      /* weight_ptr     */ weight_ptr,
      /* bias_desc      */ bias_desc.desc(),
      /* bias_ptr       */ bias_ptr,
      /* workspace_ptr  */ workspace_ptr.get(),
      /* workspace_size */ workspace_size,
      /* output_desc    */ output_desc.desc(),
      /* output_ptr     */ output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
