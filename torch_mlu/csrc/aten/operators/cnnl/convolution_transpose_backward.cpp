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

#include "aten/operators/cnnl/convolution.h"

namespace torch_mlu {
namespace ops {

// aligned with cudnn_convolution_backward.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_convolution_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params,
    std::array<bool, 3> output_mask) {
  at::Tensor grad_input, grad_weight, grad_bias;
  // memory format only support CL
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output_grad_contiguous = cnnl_contiguous(grad_output, memory_format);
  // C channel size.
  const int dim = 1;
  if (output_mask[0]) {
    at::Tensor bias;
    grad_input = cnnl_convolution_normal_forward(
        "convolution_transpose_backward_input",
        output_grad_contiguous,
        weight,
        bias,
        params);
  }
  if (output_mask[1]) {
    grad_weight = at::empty_like(weight, memory_format);
    warp_cnnl_convolution_backward_weight_internal(
        "convolution_transpose_backward_weight",
        grad_weight,
        input_contiguous,
        output_grad_contiguous,
        params);
  }
  if (output_mask[2]) {
    // compute bias grad in dim C
    grad_bias = at::empty(
        {output_grad_contiguous.size(dim)}, output_grad_contiguous.options());
    cnnl_bias_backward_internal(grad_bias, output_grad_contiguous, dim);
  }
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>{
      grad_input, grad_weight, grad_bias};
}

} // namespace ops
} // namespace torch_mlu
