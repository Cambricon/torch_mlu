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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_reflection_pad2d_backward_out_template(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  if (grad_input.numel() == 0) {
    return;
  }

  int plane_dim = 0;
  int dim_h = 1;
  int dim_w = 2;
  int nbatch = 1;

  if (input.ndimension() == 4) {
    nbatch = input.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  /* sizes */
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      output_w == grad_output.size(dim_w),
      "gradOutput width unexpected. Expected: ",
      output_w,
      ", Got: ",
      grad_output.size(dim_w));

  TORCH_CHECK(
      output_h == grad_output.size(dim_h),
      "gradOutput height unexpected. Expected: ",
      output_h,
      ", Got: ",
      grad_output.size(dim_h));

  TORCH_CHECK(
      pad_l >= 0 && pad_r >= 0 && pad_t >= 0 && pad_b >= 0,
      "negative padding values are not supported now");

  auto grad_output_contiguous = cnnl_contiguous(grad_output);
  auto grad_input_contiguous = cnnl_contiguous(grad_input);
  cnnl_reflection_pad2d_backward_internal(
      grad_input_contiguous, grad_output_contiguous, padding);
  if (!grad_input.is_same(grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
}

at::Tensor& cnnl_reflection_pad2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  cnnl_reflection_pad2d_backward_out_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

at::Tensor cnnl_reflection_pad2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  cnnl_reflection_pad2d_backward_out_template(
      grad_input, grad_output, input, padding);
  return grad_input;
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_mlu)
(const at::Tensor& grad_output,
 const at::Tensor& input,
 at::IntArrayRef padding,
 const at::Tensor& grad_input) {
  grad_input.zero_();
  if (grad_input.numel() == 0) {
    return;
  }
  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_plane++;
    dim_w++;
  }

  auto pad_l = padding[0];
  auto pad_r = padding[1];
  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  // TODO(tdai): CNNL can support negative padding values for
  // reflection_pad1d_backward, but now we use 2d impl to realize 1d,
  // because 1d requires for NLC layout that bring transpose.
  // reflection_pad2d_backward can not support negative padding values.
  TORCH_CHECK(
      pad_l >= 0 && pad_r >= 0,
      "negative padding values are not supported now");

  auto grad_input_contiguous = cnnl_contiguous(grad_input);
  auto grad_output_contiguous = cnnl_contiguous(grad_output);
  cnnl_reflection_pad1d_backward_internal(
      grad_input_contiguous, grad_output_contiguous, padding);
  if (!grad_input.is_same(grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
}

} // namespace ops
} // namespace torch_mlu
