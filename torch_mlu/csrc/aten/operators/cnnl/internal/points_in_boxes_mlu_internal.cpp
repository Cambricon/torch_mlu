#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_points_in_boxes_mlu_internal(
    at::Tensor& point_indices,
    const at::Tensor& points,
    const at::Tensor& boxes) {
  // get current handle
  auto handle = getCurrentHandle();

  // get cnnl descriptor
  CnnlTensorDescriptor points_desc;
  CnnlTensorDescriptor boxes_desc;
  CnnlTensorDescriptor output_desc;
  points_desc.set(points, CNNL_LAYOUT_ARRAY);
  boxes_desc.set(boxes, CNNL_LAYOUT_ARRAY);
  output_desc.set(point_indices, CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto points_impl = getMluTensorImpl(points);
  auto boxes_impl = getMluTensorImpl(boxes);
  auto output_impl = getMluTensorImpl(point_indices);
  auto points_ptr = mlu_data_ptr(points_impl);
  auto boxes_ptr = mlu_data_ptr(boxes_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  TORCH_CNNL_CHECK(cnnlPointsInBoxes(
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
