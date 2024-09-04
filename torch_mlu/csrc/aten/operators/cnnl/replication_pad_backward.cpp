/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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
#include "aten/utils/cnnl_util.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {
TORCH_IMPL_FUNC(replication_pad1d_backward_out_mlu)
(const at::Tensor& grad_output,
 const at::Tensor& input,
 at::IntArrayRef padding,
 const at::Tensor& grad_input) {
  auto input_ = input.ndimension() == 2 ? input.unsqueeze(0) : input;
  auto grad_input_ = at::empty_like(input_);
  auto grad_output_ =
      input.ndimension() == 2 ? grad_output.unsqueeze(0) : grad_output;
  TORCH_CHECK(padding.size() == 2, "padding size is expected to be 2");
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    nbatch = input.size(0);
    (void)nbatch;
    dimw++;
    dimslices++;
  }
  /* sizes */
  int64_t iwidth = input.size(dimw);
  int64_t owidth = iwidth + pad_l + pad_r;
  TORCH_CHECK(
      owidth == grad_output.size(dimw),
      "gradOutput width unexpected. Expected: ",
      owidth,
      " Got: ",
      grad_output.size(dimw));
  if (grad_input.numel() == 0) {
    return;
  }

  // only support CNNL_LAYOUT_NCL
  at::Tensor grad_input_contiguous =
      cnnl_contiguous(grad_input_.transpose(1, 2));
  grad_input_contiguous = grad_input_contiguous.as_strided(
      grad_input_.sizes(), get_channels_last_strides_1d(grad_input_.sizes()));
  at::Tensor grad_output_contiguous =
      cnnl_contiguous(grad_output_.transpose(1, 2));
  grad_output_contiguous = grad_output_contiguous.as_strided(
      grad_output_.sizes(), get_channels_last_strides_1d(grad_output_.sizes()));
  cnnl_replication_pad1d_backward_internal(
      grad_input_contiguous, grad_output_contiguous, padding);
  if (input.ndimension() == 2) {
    grad_input_contiguous.squeeze_(0);
  }
  if (is_copy_necessary(grad_input, grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
}

at::Tensor& cnnl_replication_pad2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  int pad_l = padding[0];
  int pad_r = padding[1];
  int pad_t = padding[2];
  int pad_b = padding[3];
  int dimw = 2;
  int dimh = 1;
  int dimslices = 0;
  int64_t nbatch = 1;

  if (input.dim() == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth = iwidth + pad_l + pad_r;

  TORCH_CHECK(
      owidth == grad_output.size(dimw),
      "gradOutput width unexpected. Expected: ",
      owidth,
      ", Got: ",
      grad_output.size(dimw));
  TORCH_CHECK(
      oheight == grad_output.size(dimh),
      "gradOutput height unexpected. Expected: ",
      oheight,
      ", Got: ",
      grad_output.size(dimh));
  grad_input.resize_as_(input);
  if (grad_input.numel() == 0) {
    return grad_input;
  }

  auto grad_input_ = grad_input;
  auto grad_output_ = grad_output;
  if (input.ndimension() == 3) {
    grad_input_ = grad_input.unsqueeze(0);
    grad_output_ = grad_output.unsqueeze(0);
  }

  // cnnlReplicationPadBackward only support CNNL_LAYOUT_NHWC
  auto grad_output_contigous =
      cnnl_contiguous(grad_output_, at::MemoryFormat::ChannelsLast);
  auto grad_input_contiguous =
      cnnl_contiguous(grad_input_, at::MemoryFormat::ChannelsLast);
  cnnl_replication_pad2d_backward_internal(
      grad_input_contiguous, grad_output_contigous, padding);
  if (input.ndimension() == 3) {
    grad_input_contiguous.squeeze_(0);
  }
  if (is_copy_necessary(grad_input, grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }
  return grad_input;
}

at::Tensor cnnl_replication_pad2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  auto input_ = input.ndimension() == 3 ? input.unsqueeze(0) : input;
  auto grad_input = at::empty_like(input_, at::MemoryFormat::ChannelsLast);
  grad_input = cnnl_replication_pad2d_backward_out(
      grad_output, input, padding, grad_input);
  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
