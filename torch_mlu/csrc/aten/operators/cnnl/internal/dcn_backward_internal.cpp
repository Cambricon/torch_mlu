#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl_dcn_backward_internal(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int* padding,
    int* stride,
    int* dilation,
    int64_t deformable_group,
    int64_t conv_group,
    int64_t im2col_step,
    std::array<bool, 2> grad_input_mask,
    bool use_mask,
    int bitwidth) {
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_option = input.options().dtype(input.scalar_type());
  auto offset_option = offset.options().dtype(input.scalar_type());
  auto weight_option = weight.options().dtype(input.scalar_type());
  auto grad_input = at::empty(input.sizes(), input_option, memory_format);
  auto grad_offset = at::empty(offset.sizes(), offset_option, memory_format);
  auto grad_weight = at::empty(weight.sizes(), weight_option, memory_format);
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_offset_impl = getMluTensorImpl(grad_offset);
  auto grad_weight_impl = getMluTensorImpl(grad_weight);
  auto input_impl = getMluTensorImpl(input);
  auto offset_impl = getMluTensorImpl(offset);
  auto weight_impl = getMluTensorImpl(weight);
  CnnlTensorDescriptor grad_desc;
  CnnlTensorDescriptor grad_input_desc;
  CnnlTensorDescriptor grad_offset_desc;
  CnnlTensorDescriptor grad_mask_desc;
  CnnlTensorDescriptor grad_weight_desc;
  CnnlTensorDescriptor grad_bias_desc;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor offset_desc;
  CnnlTensorDescriptor mask_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor bias_desc;
  CnnlDCNDescriptor dcn_desc;
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
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC; // only support NHWC
  desc_set(grad, grad_desc, layout);
  desc_set(grad_input, grad_input_desc, layout);
  desc_set(grad_offset, grad_offset_desc, layout);
  desc_set(grad_weight, grad_weight_desc, layout);
  desc_set(input, input_desc, layout);
  desc_set(offset, offset_desc, layout);
  desc_set(weight, weight_desc, layout);
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
      grad_desc.desc(), getCnnlDataType(grad.dtype()));
  cnnlSetTensorDescriptorOnchipDataType(
      input_desc.desc(), getCnnlDataType(input.dtype()));
  cnnlSetTensorDescriptorOnchipDataType(
      weight_desc.desc(), getCnnlDataType(weight.dtype()));
  // prepare mask desc
  void* mask_ptr = nullptr;
  void* grad_mask_ptr = nullptr;
  at::Tensor grad_mask;
  if (use_mask) {
    grad_mask = at::empty(
        mask.sizes(), mask.options().dtype(input.scalar_type()), memory_format);
    desc_set(mask, mask_desc, layout);
    desc_set(grad_mask, grad_mask_desc, layout);
    auto mask_impl = getMluTensorImpl(mask);
    auto grad_mask_impl = getMluTensorImpl(grad_mask);
    mask_ptr = mask_impl->mlu_data_ptr();
    grad_mask_ptr = grad_mask_impl->mlu_data_ptr();
  }
  // prepare bias desc
  void* bias_ptr = nullptr;
  void* grad_bias_ptr = nullptr;
  at::Tensor grad_bias;
  // bias Tensor size need be same with grad Tensor channel size.
  if (bias.defined() && bias.dim() == 1 && bias.size(0) == grad.size(1)) {
    grad_bias =
        at::empty(bias.sizes(), bias.options().dtype(input.scalar_type()));
    bias_desc.set(bias, CNNL_LAYOUT_ARRAY);
    grad_bias_desc.set(grad_bias, CNNL_LAYOUT_ARRAY);
    auto bias_impl = getMluTensorImpl(bias);
    auto grad_bias_impl = getMluTensorImpl(grad_bias);
    bias_ptr = bias_impl->mlu_data_ptr();
    grad_bias_ptr = grad_bias_impl->mlu_data_ptr();
  }
  // malloc mlu memory
  void* grad_ptr = grad_impl->mlu_data_ptr();
  void* grad_input_ptr = grad_input_impl->mlu_data_ptr();
  void* grad_offset_ptr = grad_offset_impl->mlu_data_ptr();
  void* grad_weight_ptr = grad_weight_impl->mlu_data_ptr();
  void* input_ptr = input_impl->mlu_data_ptr();
  void* offset_ptr = offset_impl->mlu_data_ptr();
  void* weight_ptr = weight_impl->mlu_data_ptr();
  if (grad_input_mask[0]) {
    size_t data_workspace_size = 0;
    TORCH_CNNL_CHECK(cnnlGetDCNBakcwardDataWorkspaceSize(
        /* handle           */ handle,
        /* dcn_desc         */ dcn_desc.desc(),
        /* input_desc       */ input_desc.desc(),
        /* offset_desc      */ offset_desc.desc(),
        /* mask_desc        */ mask_desc.desc(),
        /* weight_desc      */ weight_desc.desc(),
        /* grad_desc        */ grad_desc.desc(),
        /* grad_input_desc  */ grad_input_desc.desc(),
        /* grad_offset_desc */ grad_offset_desc.desc(),
        /* grad_mask_desc   */ grad_mask_desc.desc(),
        /* workspace_size   */ &data_workspace_size));
    // mallc data workspace mlu memory
    auto data_workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(data_workspace_size);
    TORCH_CNNL_CHECK(cnnlDCNBackwardData(
        /* handle           */ handle,
        /* dcn_desc         */ dcn_desc.desc(),
        /* input_desc       */ input_desc.desc(),
        /* input_ptr        */ input_ptr,
        /* offset_desc      */ offset_desc.desc(),
        /* offset_ptr       */ offset_ptr,
        /* mask_desc        */ mask_desc.desc(),
        /* mask_ptr         */ mask_ptr,
        /* weight_desc      */ weight_desc.desc(),
        /* weight_ptr       */ weight_ptr,
        /* grad_output_desc */ grad_desc.desc(),
        /* grad_output_ptr  */ grad_ptr,
        /* workspace_ptr    */ data_workspace_ptr.get(),
        /* workspace_size   */ data_workspace_size,
        /* grad_input_desc  */ grad_input_desc.desc(),
        /* grad_input_ptr   */ grad_input_ptr,
        /* grad_offset_desc */ grad_offset_desc.desc(),
        /* grad_offset_ptr  */ grad_offset_ptr,
        /* grad_mask_desc   */ grad_mask_desc.desc(),
        /* grad_maks_ptr    */ grad_mask_ptr));
  }
  if (grad_input_mask[1]) {
    // DNCBackwardWeight
    size_t weight_workspace_size = 0;
    TORCH_CNNL_CHECK(cnnlGetDCNBackwardWeightWorkspaceSize(
        /* handle            */ handle,
        /* dcn_desc          */ dcn_desc.desc(),
        /* input_desc        */ input_desc.desc(),
        /* offset_desc       */ offset_desc.desc(),
        /* mask_desc         */ mask_desc.desc(),
        /* grad_output_desc  */ grad_desc.desc(),
        /* grad_weight_desc  */ grad_weight_desc.desc(),
        /* grad_bias_desc    */ grad_bias_desc.desc(),
        /* workspace_size    */ &weight_workspace_size));
    // malloc weight workspace mlu memory
    auto weight_workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(weight_workspace_size);
    TORCH_CNNL_CHECK(cnnlDCNBackwardWeight(
        /* handle            */ handle,
        /* dcn_desc          */ dcn_desc.desc(),
        /* input_desc        */ input_desc.desc(),
        /* input_ptr         */ input_ptr,
        /* offset_desc       */ offset_desc.desc(),
        /* offset_ptr        */ offset_ptr,
        /* mask_desc         */ mask_desc.desc(),
        /* mask_ptr          */ mask_ptr,
        /* grad_output_desc  */ grad_desc.desc(),
        /* grad_output_ptr   */ grad_ptr,
        /* workspace         */ weight_workspace_ptr.get(),
        /* workspace_size    */ weight_workspace_size,
        /* grad_weight_desc  */ grad_weight_desc.desc(),
        /* grad_weigth_ptr   */ grad_weight_ptr,
        /* grad_bias_desc    */ grad_bias_desc.desc(),
        /* grad_bias_ptr     */ grad_bias_ptr));
  }
  if (!use_mask) {
    grad_mask = at::zeros_like(mask.contiguous());
  }
  auto output = std::make_tuple(
      grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
  return output;
}

} // namespace ops
} // namespace torch_mlu
