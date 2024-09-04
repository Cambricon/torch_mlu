#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_roi_align_internal(
    at::Tensor& output, // output feature map.
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    const double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    const int64_t pooled_height, // The height of the pooled feature map.
    const int64_t pooled_width, // The width of the pooled feature.
    const int64_t sampling_ratio, // The number of points to sample in each bin.
    const bool aligned) { // The flag for pixel shift.

  // get tensor impl
  auto self_impl = getMluTensorImpl(input);
  auto desc_self = getTensorDesc(self_impl, CNNL_LAYOUT_NHWC);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto rois_impl = getMluTensorImpl(rois);
  auto desc_rois = getTensorDesc(rois_impl, CNNL_LAYOUT_ARRAY);
  auto rois_ptr = mlu_data_ptr(rois_impl);

  auto output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_NHWC);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  CnnlRoiAlignDescriptor desc_roialign;
  // pytorch use avg mode
  int pool_mode = 1;
  desc_roialign.set(
      pooled_height,
      pooled_width,
      sampling_ratio,
      spatial_scale,
      pool_mode,
      aligned);

  // compute ops
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "mlu_roi_align_forward",
      [&] {
        TORCH_CNNL_CHECK(cnnlRoiAlign_v2(
            handle,
            desc_roialign.desc(),
            desc_self.get(),
            self_ptr,
            desc_rois.get(),
            rois_ptr,
            desc_output.get(),
            output_ptr,
            // pytorch not support max mode
            nullptr,
            nullptr,
            nullptr,
            nullptr));
      });

  return output;
}

at::Tensor& cnnl_roi_align_backward_internal(
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
    const bool aligned,
    at::Tensor& grad_input) {
  // get tensor impl
  auto grad_impl = getMluTensorImpl(grad);
  auto desc_grad = getTensorDesc(grad_impl, CNNL_LAYOUT_NHWC);
  auto grad_ptr = mlu_data_ptr(grad_impl);

  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto desc_grad_input = getTensorDesc(grad_input_impl, CNNL_LAYOUT_NHWC);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);

  auto rois_impl = getMluTensorImpl(rois);
  auto desc_rois = getTensorDesc(rois_impl, CNNL_LAYOUT_ARRAY);
  auto rois_ptr = mlu_data_ptr(rois_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // compute ops
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "mlu_roi_align_backward",
      [&] {
        TORCH_CNNL_CHECK(cnnlRoiAlignBackward(
            handle,
            spatial_scale,
            sampling_ratio,
            aligned,
            desc_grad.get(),
            grad_ptr,
            desc_rois.get(),
            rois_ptr,
            desc_grad_input.get(),
            grad_input_ptr));
      });

  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
