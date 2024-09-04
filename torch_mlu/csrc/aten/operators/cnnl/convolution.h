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

#include "aten/utils/tensor_util.h"
#include "aten/operators/cnnl/convolution_utils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// aligned with cudnn_convolution_forward, but without cudnn convolution check.
at::Tensor cnnl_convolution_normal_forward(
    at::CheckedFrom c,
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    const ConvParams<int64_t>& params);

// Almost same with cudnn_convolution_backward.
std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_convolution_normal_backward(
    const at::Tensor& input,
    const at::Tensor& output_grad,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params,
    std::array<bool, 3> output_mask);

at::Tensor cnnl_convolution_transpose_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_convolution_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params,
    std::array<bool, 3> output_mask);

// Add function check in warp functions.
at::Tensor& warp_cnnl_convolution_backward_input_internal(
    at::CheckedFrom c,
    at::Tensor& input_grad,
    const at::Tensor& output_grad,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params);

// Add function check in warp functions.
at::Tensor& warp_cnnl_convolution_backward_weight_internal(
    at::CheckedFrom c,
    at::Tensor& grad_weight,
    const at::Tensor& output_grad,
    const at::Tensor& input,
    const ConvParams<int64_t>& params);

} // namespace ops
} // namespace torch_mlu
