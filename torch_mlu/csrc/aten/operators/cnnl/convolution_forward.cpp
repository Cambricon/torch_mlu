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

// aligned with cudnn_convolution_forward.
at::Tensor cnnl_convolution_normal_forward(
    at::CheckedFrom c,
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    const ConvParams<int64_t>& params) {
  at::TensorArg input_arg{input, "input", 1}, weight_arg{weight, "weight", 2};
  at::checkAllSameType(c, {input_arg, weight_arg});
  checkAllSameMLU(c, {input_arg, weight_arg});
  const at::Tensor& bias = *at::borrow_from_optional_tensor(bias_opt);

  at::Tensor bias_contiguous = bias;
  if (bias.defined() && bias.numel() != 0) {
    bias_contiguous = cnnl_contiguous(bias);
  }

  auto memory_format = get_channels_last_memory_format(input.dim());
  at::Tensor output = at::empty(
      at::native::conv_output_size(
          input.sizes(),
          weight.sizes(),
          params.padding,
          params.stride,
          params.dilation),
      input.options().memory_format(memory_format));
  if (output.numel() == 0) {
    return output;
  }
  at::TensorArg output_arg{output, "result", 0};
  at::native::convolution_shape_check(
      c,
      input_arg,
      weight_arg,
      output_arg,
      params.padding,
      params.stride,
      params.dilation,
      params.groups);

  // memory format only support CL
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  // permute weight to NHWC or HWCN by add permute op to do this.
  at::Tensor weight_contiguous;
  bool is_depth_wise = false;
  if (is_mlu_depth_wise_conv(input_contiguous, weight, params.groups)) {
    // MLU depth-wise conv only support 4 dimensions.
    // Using CF contiguous weight.
    weight_contiguous = cnnl_contiguous(weight.permute({2, 3, 1, 0}));
    is_depth_wise = true;
  } else {
    weight_contiguous = cnnl_contiguous(weight, memory_format);
  }

  cnnl_convolution_forward_internal(
      output,
      input_contiguous,
      weight_contiguous,
      bias_contiguous,
      params.padding,
      params.stride,
      params.dilation,
      params.groups,
      params.benchmark,
      params.deterministic,
      params.allow_tf32,
      is_depth_wise);

  return output;
}

} // namespace ops
} // namespace torch_mlu
