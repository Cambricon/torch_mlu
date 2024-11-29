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
  auto rois_impl = getMluTensorImpl(rois);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();

  // create the desc
  CnnlTensorDescriptor desc_self;
  CnnlTensorDescriptor desc_rois;
  CnnlTensorDescriptor desc_output;
  desc_self.set(input, CNNL_LAYOUT_NHWC);
  desc_rois.set(rois, CNNL_LAYOUT_ARRAY);
  desc_output.set(output, CNNL_LAYOUT_NHWC);
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

  // get the mlu ptr
  auto self_ptr = self_impl->mlu_data_ptr();
  auto rois_ptr = rois_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

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
            desc_self.desc(),
            self_ptr,
            desc_rois.desc(),
            rois_ptr,
            desc_output.desc(),
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
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto rois_impl = getMluTensorImpl(rois);

  // get current handle
  auto handle = getCurrentHandle();

  // create the desc
  CnnlTensorDescriptor desc_grad;
  CnnlTensorDescriptor desc_grad_input;
  CnnlTensorDescriptor desc_rois;
  desc_grad.set(grad, CNNL_LAYOUT_NHWC);
  desc_grad_input.set(grad_input, CNNL_LAYOUT_NHWC);
  desc_rois.set(rois, CNNL_LAYOUT_ARRAY);

  // get the mlu ptr
  auto grad_ptr = grad_impl->mlu_data_ptr();
  auto rois_ptr = rois_impl->mlu_data_ptr();
  auto grad_input_ptr = grad_input_impl->mlu_data_ptr();

  // compute ops
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "mlu_roi_align_backward",
      [&] {
        TORCH_CNNL_CHECK(cnnlRoiAlignBackward_v2(
            handle,
            desc_grad.desc(),
            grad_ptr,
            desc_rois.desc(),
            rois_ptr,
            // pytorch not support max mode
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            spatial_scale,
            sampling_ratio,
            aligned,
            1,
            desc_grad_input.desc(),
            grad_input_ptr));
      });

  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
