#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_roi_align(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    const double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    const int64_t pooled_height, // The height of the pooled feature map.
    const int64_t pooled_width, // The width of the pooled feature.
    const int64_t sampling_ratio, // The number of points to sample in each bin.
    const bool aligned) { // The flag for pixel shift.

  TORCH_CHECK(input.device().is_privateuseone(), "input must be a MLU tensor");
  TORCH_CHECK(rois.device().is_privateuseone(), "rois must be a MLU tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "cnnl_roi_align";
  at::checkAllSameType(c, {input_t, rois_t});

  torch_mlu::mlu::MLUGuard guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_ = cnnl_contiguous(input, memory_format);
  at::Tensor output = at::empty(
      {num_rois, channels, pooled_height, pooled_width},
      input.options(),
      memory_format);
  auto rois_ = cnnl_contiguous(rois);

  if (output.numel() == 0) {
    output = at::zeros(
        {num_rois, channels, pooled_height, pooled_width}, input.options());
    return output;
  }

  cnnl_roi_align_internal(
      output,
      input_,
      rois_,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
  return output;
}

at::Tensor cnnl__roi_align_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t sampling_ratio,
    const bool aligned) {
  TORCH_CHECK(grad.device().is_privateuseone(), "grad must be a MLU tensor");
  TORCH_CHECK(rois.device().is_privateuseone(), "rois must be a MLU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "cnnl_roi_align_backward";
  at::checkAllSameType(c, {grad_t, rois_t});

  torch_mlu::mlu::MLUGuard guard(grad.device());

  auto memory_format = get_channels_last_memory_format(grad.dim());
  auto grad_ = cnnl_contiguous(grad, memory_format);
  at::Tensor grad_input = at::empty(
      {batch_size, channels, height, width}, grad.options(), memory_format);
  auto rois_ = cnnl_contiguous(rois);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    grad_input =
        at::zeros({batch_size, channels, height, width}, grad.options());
    return grad_input;
  }

  cnnl_roi_align_backward_internal(
      grad_,
      rois_,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio,
      aligned,
      grad_input);

  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
