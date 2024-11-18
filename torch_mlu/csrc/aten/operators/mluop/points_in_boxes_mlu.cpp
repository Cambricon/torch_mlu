#include "aten/operators/mluop/mluop_kernel.h"
#include "aten/operators/mluop/internal/mluop_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor mluop_points_in_boxes_mlu(
    const at::Tensor& points,
    const at::Tensor& boxes) {
  TORCH_MLU_CHECK(
      points.scalar_type() == at::ScalarType::Float,
      "Only float type is supported for points, in points_in_boxes_mlu");
  TORCH_MLU_CHECK(
      boxes.scalar_type() == at::ScalarType::Float,
      "Only float type is supported for boxes, in points_in_boxes_mlu.");

  auto points_shape = points.sizes().vec();
  auto boxes_shape = boxes.sizes().vec();
  TORCH_MLU_CHECK(
      points_shape.size() == 3,
      "Dimension of points must be equal to 3, in points_in_boxes_mlu");
  TORCH_MLU_CHECK(
      points_shape[2] == 3,
      "The last dimension of points must be equal to 3, in points_in_boxes_mlu");
  TORCH_MLU_CHECK(
      boxes_shape.size() == 3,
      "Dimension of boxes must be equal to 3, in points_in_boxes_mlu");
  TORCH_MLU_CHECK(
      boxes_shape[2] == 7,
      "The last dimension of boxes must be equal to 7, in points_in_boxes_mlu");
  TORCH_MLU_CHECK(
      boxes_shape[0] == points_shape[0],
      "The first dimension of points and boxes must be equal, in points_in_boxes_mlu");

  at::TensorArg points_arg{points, "output", 1};
  at::TensorArg boxes_arg{boxes, "indices", 2};
  checkAllSameMLU("cnnl_points_in_boxes_mlu", {points_arg, boxes_arg});

  torch_mlu::mlu::MLUGuard guard(points.device());

  auto point_indices = at::empty(
      {points_shape[0], points_shape[1]},
      points.options().dtype(at::ScalarType::Int));
  if (points.numel() == 0 || boxes.numel() == 0) {
    return point_indices.fill_(-1);
  }
  auto points_contiguous = cnnl_contiguous(points);
  auto boxes_contiguous = cnnl_contiguous(boxes);
  mluop_points_in_boxes_mlu_internal(
      point_indices, points_contiguous, boxes_contiguous);
  return point_indices;
}

} // namespace ops
} // namespace torch_mlu
