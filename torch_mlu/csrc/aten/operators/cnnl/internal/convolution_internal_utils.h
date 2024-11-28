/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <vector>
#include <tuple>
#include "ATen/Tensor.h"
#include "aten/cnnl/cnnlOpDescriptors.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

inline bool is_can_coalesce_second_dim(
    const at::IntArrayRef& weight_size,
    const int input_dim,
    const int64_t* padding,
    const int64_t* stride,
    const int64_t* dilation) {
  return input_dim == 5 && weight_size[2] == 1 && stride[0] == 1 &&
      padding[0] == 0 && dilation[0] == 1;
}

// Dont add more mlu check for this function, so keep it force inline
// to where it be called. This is only used when conv3d convert to conv2d,
// and which is satisfied func is_can_coalesce_second_dim.
inline void coalesce_conv_second_dim(
    const at::Tensor& self,
    cnnlDataType_t data_type,
    tensorDescPtr_t& desc,
    std::vector<int64_t>& shape) {
  auto combine_second_dims = [](const at::IntArrayRef& sizes,
                                std::vector<int64_t>& output) -> void {
    output[0] = sizes[0] * sizes[2];
    output[1] = sizes[1];
    output[2] = sizes[3];
    output[3] = sizes[4];
  };
  combine_second_dims(self.sizes(), shape);
  auto stride = std::move(get_channels_last_strides(shape));
  desc = getTensorDesc(shape, stride, data_type, CNNL_LAYOUT_NHWC);
}

inline bool is_can_coalesce_last_dim(
    const at::IntArrayRef& weight_size,
    const int input_dim,
    const int64_t* padding,
    const int64_t* stride,
    const int64_t* dilation) {
  return input_dim == 5 && weight_size[3] == 1 && weight_size[4] == 1 &&
      stride[1] == 1 && stride[2] == 1 && padding[1] == 0 && padding[2] == 0 &&
      dilation[1] == 1 && dilation[2] == 1;
}

// Dont add more mlu check for this function, so keep it force inline
// to where it be called. This is only used when conv3d convert to conv2d,
// and which is satisfied func is_can_coalesce_second_dim.
inline void coalesce_conv_last_dim(
    const at::Tensor& self,
    cnnlDataType_t data_type,
    tensorDescPtr_t& desc,
    std::vector<int64_t>& shape) {
  auto combine_last_dims = [](const at::IntArrayRef& sizes,
                              std::vector<int64_t>& output) -> void {
    output[0] = sizes[0];
    output[1] = sizes[1];
    output[2] = sizes[2];
    output[3] = sizes[3] * sizes[4];
  };
  combine_last_dims(self.sizes(), shape);
  auto stride = std::move(get_channels_last_strides(shape));
  desc = getTensorDesc(shape, stride, data_type, CNNL_LAYOUT_NHWC);
}

} // namespace ops
} // namespace torch_mlu
