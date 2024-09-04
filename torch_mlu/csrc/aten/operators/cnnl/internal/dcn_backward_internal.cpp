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

  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC; // only support NHWC

  auto grad_impl = getMluTensorImpl(grad);
  auto grad_desc = getTensorDesc(grad_impl, layout);
  void* grad_ptr = mlu_data_ptr(grad_impl);

  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_input_desc = getTensorDesc(grad_input_impl, layout);
  void* grad_input_ptr = mlu_data_ptr(grad_input_impl);

  auto grad_offset_impl = getMluTensorImpl(grad_offset);
  auto grad_offset_desc = getTensorDesc(grad_offset_impl, layout);
  void* grad_offset_ptr = mlu_data_ptr(grad_offset_impl);

  auto grad_weight_impl = getMluTensorImpl(grad_weight);
  auto grad_weight_desc = getTensorDesc(grad_weight_impl, layout);
  void* grad_weight_ptr = mlu_data_ptr(grad_weight_impl);

  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl, layout);
  void* input_ptr = mlu_data_ptr(input_impl);

  auto offset_impl = getMluTensorImpl(offset);
  auto offset_desc = getTensorDesc(offset_impl, layout);
  void* offset_ptr = mlu_data_ptr(offset_impl);

  auto weight_impl = getMluTensorImpl(weight);
  auto weight_desc = getTensorDesc(weight_impl, layout);
  void* weight_ptr = mlu_data_ptr(weight_impl);

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
  tensorDescPtr_t grad_mask_desc;
  tensorDescPtr_t mask_desc;
  void* mask_ptr = nullptr;
  void* grad_mask_ptr = nullptr;
  at::Tensor grad_mask;
  if (use_mask) {
    grad_mask = at::empty(
        mask.sizes(), mask.options().dtype(input.scalar_type()), memory_format);

    auto mask_impl = getMluTensorImpl(mask);
    mask_desc = getTensorDesc(mask_impl, layout);
    mask_ptr = mlu_data_ptr(mask_impl);

    auto grad_mask_impl = getMluTensorImpl(grad_mask);
    grad_mask_desc = getTensorDesc(grad_mask_impl, layout);
    grad_mask_ptr = mlu_data_ptr(grad_mask_impl);
  }

  tensorDescPtr_t grad_bias_desc;
  tensorDescPtr_t bias_desc;
  // prepare bias desc
  void* bias_ptr = nullptr;
  void* grad_bias_ptr = nullptr;
  at::Tensor grad_bias;
  // bias Tensor size need be same with grad Tensor channel size.
  if (bias.defined() && bias.dim() == 1 && bias.size(0) == grad.size(1)) {
    grad_bias =
        at::empty(bias.sizes(), bias.options().dtype(input.scalar_type()));

    auto bias_impl = getMluTensorImpl(bias);
    bias_desc = getTensorDesc(bias_impl, CNNL_LAYOUT_ARRAY);
    bias_ptr = mlu_data_ptr(bias_impl);

    auto grad_bias_impl = getMluTensorImpl(grad_bias);
    grad_bias_desc = getTensorDesc(grad_bias_impl, CNNL_LAYOUT_ARRAY);
    grad_bias_ptr = mlu_data_ptr(grad_bias_impl);
  }

  // get current handle
  auto handle = getCurrentHandle();

  if (grad_input_mask[0]) {
    size_t data_workspace_size = 0;
    TORCH_CNNL_CHECK(cnnlGetDCNBakcwardDataWorkspaceSize(
        /* handle           */ handle,
        /* dcn_desc         */ dcn_desc.desc(),
        /* input_desc       */ input_desc.get(),
        /* offset_desc      */ offset_desc.get(),
        /* mask_desc        */ mask_desc.get(),
        /* weight_desc      */ weight_desc.get(),
        /* grad_desc        */ grad_desc.get(),
        /* grad_input_desc  */ grad_input_desc.get(),
        /* grad_offset_desc */ grad_offset_desc.get(),
        /* grad_mask_desc   */ grad_mask_desc.get(),
        /* workspace_size   */ &data_workspace_size));
    // mallc data workspace mlu memory
    auto data_workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(data_workspace_size);
    TORCH_CNNL_CHECK(cnnlDCNBackwardData(
        /* handle           */ handle,
        /* dcn_desc         */ dcn_desc.desc(),
        /* input_desc       */ input_desc.get(),
        /* input_ptr        */ input_ptr,
        /* offset_desc      */ offset_desc.get(),
        /* offset_ptr       */ offset_ptr,
        /* mask_desc        */ mask_desc.get(),
        /* mask_ptr         */ mask_ptr,
        /* weight_desc      */ weight_desc.get(),
        /* weight_ptr       */ weight_ptr,
        /* grad_output_desc */ grad_desc.get(),
        /* grad_output_ptr  */ grad_ptr,
        /* workspace_ptr    */ data_workspace_ptr.get(),
        /* workspace_size   */ data_workspace_size,
        /* grad_input_desc  */ grad_input_desc.get(),
        /* grad_input_ptr   */ grad_input_ptr,
        /* grad_offset_desc */ grad_offset_desc.get(),
        /* grad_offset_ptr  */ grad_offset_ptr,
        /* grad_mask_desc   */ grad_mask_desc.get(),
        /* grad_maks_ptr    */ grad_mask_ptr));
  }
  if (grad_input_mask[1]) {
    // DNCBackwardWeight
    size_t weight_workspace_size = 0;
    TORCH_CNNL_CHECK(cnnlGetDCNBackwardWeightWorkspaceSize(
        /* handle            */ handle,
        /* dcn_desc          */ dcn_desc.desc(),
        /* input_desc        */ input_desc.get(),
        /* offset_desc       */ offset_desc.get(),
        /* mask_desc         */ mask_desc.get(),
        /* grad_output_desc  */ grad_desc.get(),
        /* grad_weight_desc  */ grad_weight_desc.get(),
        /* grad_bias_desc    */ grad_bias_desc.get(),
        /* workspace_size    */ &weight_workspace_size));
    // malloc weight workspace mlu memory
    auto weight_workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(weight_workspace_size);
    TORCH_CNNL_CHECK(cnnlDCNBackwardWeight(
        /* handle            */ handle,
        /* dcn_desc          */ dcn_desc.desc(),
        /* input_desc        */ input_desc.get(),
        /* input_ptr         */ input_ptr,
        /* offset_desc       */ offset_desc.get(),
        /* offset_ptr        */ offset_ptr,
        /* mask_desc         */ mask_desc.get(),
        /* mask_ptr          */ mask_ptr,
        /* grad_output_desc  */ grad_desc.get(),
        /* grad_output_ptr   */ grad_ptr,
        /* workspace         */ weight_workspace_ptr.get(),
        /* workspace_size    */ weight_workspace_size,
        /* grad_weight_desc  */ grad_weight_desc.get(),
        /* grad_weigth_ptr   */ grad_weight_ptr,
        /* grad_bias_desc    */ grad_bias_desc.get(),
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
