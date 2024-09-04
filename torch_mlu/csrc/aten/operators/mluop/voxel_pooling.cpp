#include "aten/operators/mluop/mluop_kernel.h"
#include "aten/operators/mluop/internal/mluop_internal.h"

namespace torch_mlu {
namespace ops {

bool mluop_voxel_pooling(
    int64_t batch_size,
    int64_t num_points,
    int64_t num_channels,
    int64_t num_voxel_x,
    int64_t num_voxel_y,
    int64_t num_voxel_z,
    const at::Tensor& geom_xyz,
    const at::Tensor& input_features,
    at::Tensor& output_features,
    at::Tensor& pos_memo) {
  TORCH_MLU_CHECK(
      (batch_size > 0 && num_points > 0 && num_channels > 0 &&
       num_voxel_x > 0 && num_voxel_y > 0 && num_voxel_z > 0),
      "batch_size, num_points, num_channels, num_voxel_x, num_voxel_y must be positive, but got ",
      batch_size,
      ", ",
      num_points,
      ", ",
      num_channels,
      ", ",
      num_voxel_x,
      ", ",
      num_voxel_y,
      ", ",
      num_voxel_z);
  TORCH_MLU_CHECK(
      geom_xyz.device().is_privateuseone(), "geom_xyz must be a MLU tensor.");
  TORCH_MLU_CHECK(
      input_features.device().is_privateuseone(),
      "input_features must be a MLU tensor.");
  TORCH_MLU_CHECK(
      geom_xyz.scalar_type() == at::ScalarType::Int &&
          pos_memo.scalar_type() == at::ScalarType::Int,
      "geom_xyz and pos_memo must be int tensor.");
  TORCH_MLU_CHECK(
      input_features.scalar_type() == at::ScalarType::Float &&
          output_features.scalar_type() == at::ScalarType::Float,
      "input_features and output_features must be float tensor.");

  auto geom_xyz_contiguous = torch_mlu::cnnl_contiguous(geom_xyz);
  auto input_features_contiguous = torch_mlu::cnnl_contiguous(input_features);
  mluop_voxel_pooling_internal(
      geom_xyz_contiguous,
      input_features_contiguous,
      batch_size,
      num_points,
      num_channels,
      num_voxel_x,
      num_voxel_y,
      num_voxel_z,
      output_features,
      pos_memo);

  return true;
}

} // namespace ops
} // namespace torch_mlu
