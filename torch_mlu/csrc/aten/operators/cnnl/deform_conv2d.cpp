#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_deform_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  bool areSameType = input.scalar_type() == weight.scalar_type() &&
      input.scalar_type() == offset.scalar_type() &&
      input.scalar_type() == mask.scalar_type() &&
      input.scalar_type() == bias.scalar_type();
  TORCH_CHECK(
      areSameType,
      "Expected input, weight, offset, mask, bias have same dtype");
  TORCH_CHECK(input.is_floating_point(), "only support real floating point");
  TORCH_CHECK(input.device().is_privateuseone(), "input must be a MLU tensor");
  // check input sizes
  TORCH_CHECK(
      input.dim() == 4, "deform_conv2d op input's dim must equal to 4.");
  TORCH_CHECK(
      weight.dim() == 4, "deform_conv2d op weight's dim must equal to 4.");
  TORCH_CHECK(
      offset.dim() == 4, "deform_conv2d op offset's dim must equal to 4.");
  TORCH_CHECK(
      !use_mask || mask.dim() == 4,
      "deform_conv2d op mask's dim must equal to 4, if use_mask is true.");

  int batch_sz = input.size(0);
  int n_in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  int out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int ker_h = dilation_h * (weight_h - 1) + 1;
  int ker_w = dilation_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(
      weight_h > 0 && weight_w > 0,
      "dcn_forward: 2 and 3 dim of weight should be greater than zero, but got 2 dim: ",
      weight_h,
      ", 3 dim: ",
      weight_w);
  TORCH_CHECK(
      stride_h > 0 && stride_w > 0,
      "dcn_forward: stride should be greater than zero,but got sH: ",
      stride_h,
      ", sW: ",
      stride_w);
  TORCH_CHECK(
      pad_h >= 0 && pad_w >= 0,
      "dcn_forward: pad should be greater than or equal zero, but got padLeft: ",
      pad_h,
      ", padRight: ",
      pad_w);
  TORCH_CHECK(
      dilation_h > 0 && dilation_w > 0,
      "dcn_forward: dilation should be greater than zero, but got dH: ",
      dilation_h,
      " dW: ",
      dilation_w);
  TORCH_CHECK(weight.size(1) * groups == n_in_channels);
  TORCH_CHECK(weight.size(0) % groups == 0);
  TORCH_CHECK(
      (offset.size(1) == offset_groups * 2 * weight_h * weight_w),
      "offset.shape[1] is not valid: got: ",
      offset.size(1),
      " expected: ",
      offset_groups * 2 * weight_h * weight_w);
  TORCH_CHECK(
      (!use_mask || mask.size(1) == offset_groups * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ",
      mask.size(1),
      " expected: ",
      offset_groups * weight_h * weight_w);
  TORCH_CHECK(input.size(1) % offset_groups == 0);
  TORCH_CHECK(
      (offset.size(0) == input.size(0)), "invalid batch size of offset");
  TORCH_CHECK(
      (offset.size(2) == out_h && offset.size(3) == out_w),
      "offset output dims: (",
      offset.size(2),
      ", ",
      offset.size(3),
      ") - computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK((mask.size(0) == input.size(0)), "invalid batch size of mask");
  TORCH_CHECK(
      (!use_mask || (mask.size(2) == out_h && mask.size(3) == out_w)),
      "offset output dims: (",
      mask.size(2),
      ", ",
      mask.size(3),
      ") - computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ",
      out_h,
      " out_w: ",
      out_w);

  std::vector<int64_t> output_size = {batch_sz, out_channels, out_h, out_w};
  if (batch_sz == 0) {
    return at::zeros(output_size, input.contiguous().options());
  }

  int stride_t[2] = {(int)stride_h, (int)stride_w};
  int padding_t[4] = {(int)pad_h, (int)pad_h, (int)pad_w, (int)pad_w};
  int dilation_t[2] = {(int)dilation_h, (int)dilation_w};
  int bitwidth_value = 16;
  // set tensor contiguous
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto offset_contiguous = cnnl_contiguous(offset, memory_format);
  auto mask_contiguous = use_mask ? cnnl_contiguous(mask, memory_format) : mask;
  auto weight_contiguous = cnnl_contiguous(weight, memory_format);
  auto bias_contiguous = cnnl_contiguous(bias);
  int im2col_step = batch_sz;
  auto out = cnnl_dcn_forward_internal(
      input_contiguous,
      offset_contiguous,
      mask_contiguous,
      weight_contiguous,
      bias_contiguous,
      output_size,
      padding_t,
      stride_t,
      dilation_t,
      offset_groups,
      groups,
      im2col_step,
      bitwidth_value,
      use_mask);
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__deform_conv2d_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  if (input.size(0) == 0) {
    auto grad_input = at::zeros_like(input.contiguous());
    auto grad_offset = at::zeros_like(offset.contiguous());
    auto grad_mask = at::zeros_like(mask.contiguous());
    auto grad_weight = at::zeros_like(weight.contiguous());
    auto value = grad.contiguous().sum({0, 2, 3});
    auto grad_bias = at::ones_like(bias.contiguous()) * value;
    return std::make_tuple(
        grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
  }

  int stride_t[2] = {(int)stride_h, (int)stride_w};
  int dilation_t[2] = {(int)dilation_h, (int)dilation_w};
  int padding_t[4] = {(int)pad_h, (int)pad_h, (int)pad_w, (int)pad_w};
  int bitwidth_value = 31;
  // set tensor contiguous
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto grad_contiguous = cnnl_contiguous(grad, memory_format);
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto offset_contiguous = cnnl_contiguous(offset, memory_format);
  auto mask_contiguous = use_mask ? cnnl_contiguous(mask, memory_format) : mask;
  auto weight_contiguous = cnnl_contiguous(weight, memory_format);
  auto bias_contiguous = cnnl_contiguous(bias);
  int im2col_step = input.size(0);
  std::array<bool, 2> grad_input_mask = {true, true};
  auto output = cnnl_dcn_backward_internal(
      grad_contiguous,
      input_contiguous,
      offset_contiguous,
      mask_contiguous,
      weight_contiguous,
      bias_contiguous,
      padding_t,
      stride_t,
      dilation_t,
      offset_groups,
      groups,
      im2col_step,
      grad_input_mask,
      use_mask,
      bitwidth_value);
  return output;
}

} // namespace ops

at::Tensor deform_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::deform_conv2d", "")
                       .typed<decltype(deform_conv2d)>();
  return op.call(
      input,
      weight,
      offset,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      groups,
      offset_groups,
      use_mask);
}

at::Tensor deform_conv2d_autocast(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    int64_t offset_groups,
    bool use_mask) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
      c10::DispatchKey::AutocastPrivateUse1);
  return deform_conv2d(
             at::autocast::cached_cast(
                 at::kFloat, input, DeviceType::PrivateUse1),
             at::autocast::cached_cast(
                 at::kFloat, weight, DeviceType::PrivateUse1),
             at::autocast::cached_cast(
                 at::kFloat, offset, DeviceType::PrivateUse1),
             at::autocast::cached_cast(
                 at::kFloat, mask, DeviceType::PrivateUse1),
             at::autocast::cached_cast(
                 at::kFloat, bias, DeviceType::PrivateUse1),
             stride_h,
             stride_w,
             pad_h,
             pad_w,
             dilation_h,
             dilation_w,
             groups,
             offset_groups,
             use_mask)
      .to(input.scalar_type());
}

TORCH_LIBRARY_IMPL(torchvision, AutocastPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::deform_conv2d"),
      TORCH_FN(deform_conv2d_autocast));
}

} // namespace torch_mlu
