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

// warp functions
at::Tensor& warp_cnnl_convolution_backward_input_internal(
    at::CheckedFrom c,
    at::Tensor& input_grad,
    const at::Tensor& output_grad,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params) {
  at::TensorArg input_arg{output_grad, "output_grad", 1},
      weight_arg{weight, "weight", 2}, output_arg{input_grad, "result", 0};
  at::checkAllSameType(c, {input_arg, weight_arg});
  checkAllSameMLU(c, {input_arg, weight_arg});
  at::native::convolution_shape_check(
      c,
      output_arg,
      weight_arg,
      input_arg,
      params.padding,
      params.stride,
      params.dilation,
      params.groups);
  // Call internal function
  return cnnl_convolution_backward_input_internal(
      input_grad,
      output_grad,
      weight,
      params.stride,
      params.padding,
      params.dilation,
      params.groups,
      params.benchmark,
      params.deterministic,
      params.allow_tf32);
}

at::Tensor& warp_cnnl_convolution_backward_weight_internal(
    at::CheckedFrom c,
    at::Tensor& grad_weight,
    const at::Tensor& output_grad,
    const at::Tensor& input,
    const ConvParams<int64_t>& params) {
  at::TensorArg input_arg{input, "input", 1},
      weight_arg{grad_weight, "grad_weight", 2},
      output_arg{output_grad, "result", 0};
  at::checkAllSameType(c, {input_arg, weight_arg});
  checkAllSameMLU(c, {input_arg, weight_arg});
  at::native::convolution_shape_check(
      c,
      input_arg,
      weight_arg,
      output_arg,
      params.padding,
      params.stride,
      params.dilation,
      params.groups);

  cnnl_convolution_backward_weight_internal(
      grad_weight,
      output_grad,
      input,
      params.stride,
      params.padding,
      params.dilation,
      params.groups,
      params.benchmark,
      params.deterministic,
      params.allow_tf32);
  return grad_weight;
}

// aligned with cudnn_convolution_forward.
std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_convolution_normal_backward(
    const at::Tensor& input,
    const at::Tensor& output_grad,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params,
    std::array<bool, 3> output_mask) {
  at::Tensor grad_input, grad_weight, grad_bias;
  auto memory_format = get_channels_last_memory_format(input.dim());
  // C channel size.
  const int dim = 1;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, memory_format);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, memory_format);
    }
    if (output_mask[2]) {
      grad_bias = at::zeros({input.size(dim)}, memory_format);
    }
  } else {
    // Contiguous
    auto input_contiguous = cnnl_contiguous(input, memory_format);
    auto weight_contiguous = cnnl_contiguous(weight, memory_format);
    auto grad_contiguous = cnnl_contiguous(output_grad, memory_format);

    if (output_mask[0]) {
      grad_input = at::empty_like(input_contiguous, memory_format);
      warp_cnnl_convolution_backward_input_internal(
          "cnnl_convolution_backward_input",
          grad_input,
          grad_contiguous,
          weight_contiguous,
          params);
    }
    if (output_mask[1]) {
      grad_weight = at::empty_like(weight_contiguous, memory_format);
      warp_cnnl_convolution_backward_weight_internal(
          "cnnl_convolution_backward_weight",
          grad_weight,
          grad_contiguous,
          input_contiguous,
          params);
    }
    if (output_mask[2]) {
      grad_bias =
          at::empty({grad_contiguous.size(dim)}, grad_contiguous.options());
      cnnl_bias_backward_internal(grad_bias, grad_contiguous, dim);
    }
  }
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>{
      grad_input, grad_weight, grad_bias};
}

} // namespace ops
} // namespace torch_mlu
