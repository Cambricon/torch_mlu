#include <ATen/autocast_mode.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceType.h>

#include <iostream>
#include <exception>

namespace {
using namespace at::autocast;

/*******************************
Banned functions
*******************************/

at::Tensor nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.nms.nms");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::nms", "")
                       .typed<decltype(nms)>();
  return op.call(dets, scores, iou_threshold);
}

at::Tensor nms_autocast(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
      c10::DispatchKey::AutocastPrivateUse1);
  return nms(
      at::autocast::cached_cast(at::kFloat, dets, c10::DeviceType::PrivateUse1),
      at::autocast::cached_cast(
          at::kFloat, scores, c10::DeviceType::PrivateUse1),
      iou_threshold);
}

at::Tensor roi_align(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    double spatial_scale, // The scale of the image features. ROIs will be
                          // scaled to this.
    int64_t pooled_height, // The height of the pooled feature map.
    int64_t pooled_width, // The width of the pooled feature
    int64_t sampling_ratio, // The number of points to sample in each bin
    bool aligned) { // The flag for pixel shift along each axis.
  C10_LOG_API_USAGE_ONCE("torchvision.csrc.ops.roi_align.roi_align");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_align", "")
                       .typed<decltype(roi_align)>();
  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor roi_align_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
      c10::DispatchKey::AutocastPrivateUse1);
  return roi_align(
             at::autocast::cached_cast(
                 at::kFloat, input, c10::DeviceType::PrivateUse1),
             at::autocast::cached_cast(
                 at::kFloat, rois, c10::DeviceType::PrivateUse1),
             spatial_scale,
             pooled_height,
             pooled_width,
             sampling_ratio,
             aligned)
      .to(input.scalar_type());
}

TORCH_LIBRARY_IMPL(torchvision, AutocastPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(roi_align_autocast));
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_autocast));
}

static at::Tensor binary_cross_entropy_banned(
    const at::Tensor&,
    const at::Tensor&,
    const std::optional<at::Tensor>&,
    int64_t) {
  AT_ERROR(
      "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are "
      "unsafe to autocast.\n"
      "Many models use a sigmoid layer right before the binary cross "
      "entropy layer.\n"
      "In this case, combine the two layers using "
      "torch.nn.functional.binary_cross_entropy_with_logits\n"
      "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits "
      "and BCEWithLogits are\n"
      "safe to autocast.");
}

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // lower_precision_fp
#define _KERNEL_PRIVATEUSEONE_LOW_PRECISION_FP(...) \
  KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_PRIVATEUSEONE_LOW_PRECISION_FP)
  KERNEL_PRIVATEUSEONE(cudnn_convolution, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(cudnn_convolution_transpose, lower_precision_fp)

  // fp32
#define _KERNEL_PRIVATEUSEONE_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_PRIVATEUSEONE_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_PRIVATEUSEONE_FP32_SET_OPT_DTYPE(...) \
  KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_PRIVATEUSEONE_FP32_SET_OPT_DTYPE)
  // commenting these out because they accept an explicit (not-optional) dtype,
  // and we shouldn't try to flip that even when autocasting.
  // KERNEL_PRIVATEUSEONE(norm, ScalarOpt_dtype, fp32_set_opt_dtype)
  // KERNEL_PRIVATEUSEONE(norm, ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  // KERNEL_PRIVATEUSEONE(norm, names_ScalarOpt_dim_dtype, fp32_set_opt_dtype)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this
  // policy.
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE(
      ADD_NS(norm),
      "norm.Scalar",
      at::Tensor(const at::Tensor&, const at::Scalar&),
      at::Tensor(
          const at::Tensor&, const std::optional<at::Scalar>&, at::ScalarType),
      fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE(
      ADD_NS(norm),
      "norm.ScalarOpt_dim",
      at::Tensor(
          const at::Tensor&,
          const std::optional<at::Scalar>&,
          c10::IntArrayRef,
          bool),
      at::Tensor(
          const at::Tensor&,
          const std::optional<at::Scalar>&,
          c10::IntArrayRef,
          bool,
          at::ScalarType),
      fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE(
      ADD_NS(norm),
      "norm.names_ScalarOpt_dim",
      at::Tensor(
          const at::Tensor&,
          const std::optional<at::Scalar>&,
          at::DimnameList,
          bool),
      at::Tensor(
          const at::Tensor&,
          const std::optional<at::Scalar>&,
          at::DimnameList,
          bool,
          at::ScalarType),
      fp32_append_dtype)

  // promote
#define _KERNEL_PRIVATEUSEONE_PROMOTE(...) \
  KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_PRIVATEUSEONE_PROMOTE)

  m.impl(
      TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
      TORCH_FN((&binary_cross_entropy_banned)));
}
} // namespace
