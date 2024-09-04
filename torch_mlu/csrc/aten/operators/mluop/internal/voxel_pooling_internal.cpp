#include "aten/operators/mluop/internal/mluop_internal.h"

namespace torch_mlu {
namespace ops {

void mluop_voxel_pooling_internal(
    const at::Tensor& geom_xyz,
    const at::Tensor& input_features,
    int batch_size,
    int num_points,
    int num_channels,
    int num_voxel_x,
    int num_voxel_y,
    int num_voxel_z,
    at::Tensor& output_features,
    at::Tensor& pos_memo) {
  auto geom_xyz_impl = getMluTensorImpl(geom_xyz);
  auto input_features_impl = getMluTensorImpl(input_features);
  auto output_features_impl = getMluTensorImpl(output_features);
  auto pos_memo_impl = getMluTensorImpl(pos_memo);
  MluOpTensorDescriptor geom_xyz_desc;
  MluOpTensorDescriptor input_features_desc;
  MluOpTensorDescriptor output_features_desc;
  MluOpTensorDescriptor pos_memo_desc;
  geom_xyz_desc.set(geom_xyz);
  input_features_desc.set(input_features);
  output_features_desc.set(output_features);
  pos_memo_desc.set(pos_memo);

  auto geom_xyz_ptr = geom_xyz_impl->mlu_data_ptr();
  auto input_features_ptr = input_features_impl->mlu_data_ptr();
  auto output_features_ptr = output_features_impl->mlu_data_ptr();
  auto pos_memo_ptr = pos_memo_impl->mlu_data_ptr();

  auto handle = getCurrentMluOpHandle();

  TORCH_MLUOP_CHECK(mluOpVoxelPoolingForward(
      handle,
      batch_size,
      num_points,
      num_channels,
      num_voxel_x,
      num_voxel_y,
      num_voxel_z,
      geom_xyz_desc.desc(),
      geom_xyz_ptr,
      input_features_desc.desc(),
      input_features_ptr,
      output_features_desc.desc(),
      output_features_ptr,
      pos_memo_desc.desc(),
      pos_memo_ptr));
}

} // namespace ops
} // namespace torch_mlu
