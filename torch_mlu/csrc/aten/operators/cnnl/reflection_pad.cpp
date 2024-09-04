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

void cnnl_reflection_pad2d_out_template(
    at::Tensor& output,
    const at::Tensor& input_,
    at::IntArrayRef padding) {
  int plane_dim = 0;
  int dim_h = 1;
  int dim_w = 2;
  int nbatch = 1;

  // allow dim=0 only in the batch dimension.
  bool valid_dims = input_.size(1) != 0 && input_.size(2) != 0;
  TORCH_CHECK(
      (input_.ndimension() == 3 && valid_dims) ||
          (input_.ndimension() == 4 && valid_dims && input_.size(3) != 0),
      "3D or 4D (batch mode) tensor expected for input, but got: ",
      input_);

  if (input_.ndimension() == 4) {
    nbatch = input_.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int nplane = input_.size(plane_dim);
  int input_h = input_.size(dim_h);
  int input_w = input_.size(dim_w);

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input_.sizes());

  TORCH_CHECK(
      pad_t < input_h && pad_b < input_h,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_t,
      ", ",
      pad_b,
      ") at dimension ",
      dim_h,
      " of input ",
      input_.sizes());

  int output_h = input_h + pad_t + pad_b;
  int output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      output_w >= 1 || output_h >= 1,
      "input (H: ",
      input_h,
      ", W: ",
      input_w,
      ")is too small.  Calculated "
      "output H: ",
      output_h,
      " W: ",
      output_w);

  if (input_.ndimension() == 3) {
    output.resize_({nplane, output_h, output_w});
  } else {
    output.resize_(
        {nbatch, nplane, output_h, output_w}, input_.suggest_memory_format());
  }

  if (output.numel() == 0) {
    return;
  }

  auto memory_format = input_.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(input_, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);

  cnnl_reflection_pad2d_internal(output_contiguous, input_contiguous, padding);

  if (!output.is_same(output_contiguous)) {
    output.copy_(output_contiguous);
  }
}

at::Tensor& cnnl_reflection_pad2d_out(
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::Tensor& output) {
  cnnl_reflection_pad2d_out_template(output, input, padding);
  return output;
}

at::Tensor cnnl_reflection_pad2d(
    const at::Tensor& input,
    at::IntArrayRef padding) {
  auto output = at::empty({0}, input.options());
  cnnl_reflection_pad2d_out_template(output, input, padding);
  return output;
}

TORCH_IMPL_FUNC(reflection_pad1d_out_mlu)
(const at::Tensor& input_, at::IntArrayRef padding, const at::Tensor& output) {
  if (output.numel() == 0) {
    return;
  }
  auto input_contiguous = cnnl_contiguous(input_);
  auto output_contiguous = cnnl_contiguous(output);
  cnnl_reflection_pad1d_internal(output_contiguous, input_contiguous, padding);
  if (!output.is_same(output_contiguous)) {
    output.copy_(output_contiguous);
  }
}

} // namespace ops
} // namespace torch_mlu
