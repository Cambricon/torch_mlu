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

#include <ATen/native/UpSample.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

TORCH_META_FUNC(upsample_linear1d_out_mlu)
(const at::Tensor& input,
 at::IntArrayRef output_size,
 bool align_corners,
 c10::optional<double> scales) {
  auto full_output_size =
      at::native::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(
      0,
      full_output_size,
      get_channels_last_strides_1d(full_output_size),
      input.options());
}

TORCH_META_FUNC(upsample_linear1d_backward_out_mlu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 c10::optional<double> scales) {
  auto full_output_size =
      at::native::upsample_1d_common_check(input_size, output_size);

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  set_output_raw_strided(
      0,
      input_size,
      get_channels_last_strides_1d(input_size),
      grad_output.options());
}

TORCH_IMPL_FUNC(upsample_linear1d_out_mlu)
(const at::Tensor& input,
 at::IntArrayRef output_size,
 bool align_corners,
 c10::optional<double> scales,
 const at::Tensor& output) {
  at::TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameMLU(__func__, {input_arg, output_arg});

  int output_width = output_size[0];
  int input_width = input.size(2);

  AT_ASSERT(input_width > 0 && output_width > 0);

  if (input.numel() == 0) {
    return;
  }

  // NLC input
  at::Tensor input_contiguous = cnnl_contiguous(input.transpose(1, 2));
  input_contiguous = input_contiguous.as_strided(
      input.sizes(), get_channels_last_strides_1d(input.sizes()));

  // NLC output
  auto output_contiguous = maybe_create_out(
      output,
      output.sizes(),
      get_channels_last_strides_1d(output.sizes()),
      output.options());

  // cnnl interp
  bool align_center = !align_corners;
  cnnlInterpMode_t interp_mode = CNNL_INTERP_LINEAR;
  cnnl_upsample_internal(
      output_contiguous,
      input_contiguous,
      output_size,
      align_corners,
      align_center,
      interp_mode,
      scales);
  if (!output.is_same(output_contiguous)) {
    output.copy_(output_contiguous);
  }
}

TORCH_IMPL_FUNC(upsample_linear1d_backward_out_mlu)
(const at::Tensor& grad_output,
 at::IntArrayRef output_size,
 at::IntArrayRef input_size,
 bool align_corners,
 c10::optional<double> scales,
 const at::Tensor& grad_input) {
  at::TensorArg grad_output_arg{grad_output, "grad_output", 1},
      grad_input_arg{grad_input, "grad_input", 2};
  checkAllSameMLU(__func__, {grad_output_arg, grad_input_arg});

  if (grad_input.numel() == 0) {
    return;
  }

  // grad_input
  auto grad_input_contiguous = maybe_create_out(
      grad_input,
      grad_input.sizes(),
      get_channels_last_strides_1d(grad_input.sizes()),
      grad_input.options());

  // grad_output
  at::Tensor grad_output_contiguous =
      cnnl_contiguous(grad_output.transpose(1, 2));
  grad_output_contiguous = grad_output_contiguous.as_strided(
      grad_output.sizes(), get_channels_last_strides_1d(grad_output.sizes()));

  // cnnl interp
  bool align_center = !align_corners;
  cnnlInterpMode_t interp_mode = CNNL_INTERP_LINEAR;
  cnnl_upsample_backward_internal(
      grad_input_contiguous,
      grad_output_contiguous,
      output_size,
      align_corners,
      align_center,
      interp_mode,
      scales);
  if (!grad_input.is_same(grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
}

} // namespace ops
} // namespace torch_mlu
