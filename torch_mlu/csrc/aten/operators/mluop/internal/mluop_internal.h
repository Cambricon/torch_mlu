#pragma once

#include "aten/mluop/mluopDescriptors.h"
#include "aten/mluop/mluopHandle.h"
#include "aten/mluop/mluopUtils.h"

#include "aten/utils/exceptions.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"
#include "framework/core/MLUStream.h"
#include "framework/core/tensor_impl.h"
#include "utils/cnlog.h"
#include "utils/common.h"

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
    at::Tensor& pos_memo);

const at::Tensor& mluop_fft_internal(
    at::Tensor& out,
    const at::Tensor& self,
    c10::IntArrayRef out_sizes,
    c10::IntArrayRef dim,
    bool forward,
    const float scale_factor);

} // namespace ops
} // namespace torch_mlu
