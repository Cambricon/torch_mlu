#include "aten/operators/mluop/internal/mluop_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& mluop_points_in_boxes_mlu_internal(
    at::Tensor& point_indices,
    const at::Tensor& points,
    const at::Tensor& boxes) {
  // get current handle
  auto handle = getCurrentMluOpHandle();

  auto points_impl = getMluTensorImpl(points);
  auto points_ptr = mlu_data_ptr(points_impl);

  auto boxes_impl = getMluTensorImpl(boxes);
  auto boxes_ptr = mlu_data_ptr(boxes_impl);

  auto output_impl = getMluTensorImpl(point_indices);
  auto output_ptr = mlu_data_ptr(output_impl);

  MluOpTensorDescriptor points_desc;
  MluOpTensorDescriptor boxes_desc;
  MluOpTensorDescriptor output_desc;
  points_desc.set(points);
  boxes_desc.set(boxes);
  output_desc.set(point_indices);

  TORCH_MLUOP_CHECK(mluOpPointsInBoxes(
      handle,
      points_desc.desc(),
      points_ptr,
      boxes_desc.desc(),
      boxes_ptr,
      output_desc.desc(),
      output_ptr));
  return point_indices;
}

} // namespace ops
} // namespace torch_mlu
