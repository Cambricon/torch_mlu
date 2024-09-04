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

TORCH_META_FUNC(upsample_bilinear2d_out_mlu)
(const at::Tensor& input,
 at::IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w) {
  auto full_output_size =
      at::native::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  set_output_raw_strided(
      0,
      full_output_size,
      {},
      input.options().memory_format(at::MemoryFormat::ChannelsLast));
}

TORCH_META_FUNC(upsample_bilinear2d_backward_out_mlu)
(const at::Tensor& grad_output,
 at::IntArrayRef output_size,
 at::IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w) {
  auto full_output_size =
      at::native::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ",
      grad_output.dim());

  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }

  set_output_raw_strided(
      0,
      input_size,
      {},
      grad_output.options().memory_format(at::MemoryFormat::ChannelsLast));
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_mlu)
(const at::Tensor& input,
 at::IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const at::Tensor& output) {
  at::TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameMLU(__func__, {input_arg, output_arg});

  if (input.sizes() == output.sizes()) {
    output.copy_(input);
    return;
  }

  if (input.numel() == 0) {
    return;
  }

  // NHWC input
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto input_contiguous = cnnl_contiguous(input, memory_format);

  // NHWC output
  auto output_contiguous = maybe_create_out(
      output,
      output.sizes(),
      get_channels_last_strides_2d(output.sizes()),
      output.options());

  // cnnl interp
  bool align_center = !align_corners;
  cnnlInterpMode_t interp_mode = CNNL_INTERP_BILINEAR;
  cnnl_upsample_internal(
      output_contiguous,
      input_contiguous,
      output_size,
      align_corners,
      align_center,
      interp_mode,
      scales_h,
      scales_w);
  if (!output.is_same(output_contiguous)) {
    output.copy_(output_contiguous);
  }
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_mlu)
(const at::Tensor& grad_output,
 at::IntArrayRef output_size,
 at::IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const at::Tensor& grad_input) {
  at::TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output, "grad_output", 2};
  checkAllSameMLU(__func__, {grad_output_arg, grad_input_arg});

  if (grad_input.numel() == 0) {
    return;
  }

  // NHWC grad_input, grad_output
  auto memory_format = at::MemoryFormat::ChannelsLast;
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto grad_input_contiguous = maybe_create_out(
      grad_input,
      grad_input.sizes(),
      get_channels_last_strides_2d(grad_input.sizes()),
      grad_input.options());

  cnnlInterpMode_t interp_mode = CNNL_INTERP_BILINEAR;
  bool align_center = !align_corners;

  cnnl_upsample_backward_internal(
      grad_input_contiguous,
      grad_output_contiguous,
      output_size,
      align_corners,
      align_center,
      interp_mode,
      scales_h,
      scales_w);
  if (!grad_input.is_same(grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
}

} // namespace ops
} // namespace torch_mlu
