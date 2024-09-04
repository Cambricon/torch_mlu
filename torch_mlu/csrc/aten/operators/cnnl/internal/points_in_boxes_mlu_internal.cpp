#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_points_in_boxes_mlu_internal(
    at::Tensor& point_indices,
    const at::Tensor& points,
    const at::Tensor& boxes) {
  // get current handle
  auto handle = getCurrentHandle();

  auto points_impl = getMluTensorImpl(points);
  auto points_desc = getTensorDesc(points_impl, CNNL_LAYOUT_ARRAY);
  auto points_ptr = mlu_data_ptr(points_impl);

  auto boxes_impl = getMluTensorImpl(boxes);
  auto boxes_desc = getTensorDesc(boxes_impl, CNNL_LAYOUT_ARRAY);
  auto boxes_ptr = mlu_data_ptr(boxes_impl);

  auto output_impl = getMluTensorImpl(point_indices);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  TORCH_CNNL_CHECK(cnnlPointsInBoxes(
      handle,
      points_desc.get(),
      points_ptr,
      boxes_desc.get(),
      boxes_ptr,
      output_desc.get(),
      output_ptr));
  return point_indices;
}

} // namespace ops
} // namespace torch_mlu
