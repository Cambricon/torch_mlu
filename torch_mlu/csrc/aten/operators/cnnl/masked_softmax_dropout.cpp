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

// input is also output, we cannot use the name "mask_softmax_dropout_fprop_",
// because currently codegen does support.
std::tuple<at::Tensor, at::Tensor> cnnl_mask_softmax_dropout_fprop(
    at::Tensor& input,
    const at::Tensor& mask,
    int64_t batch,
    const at::Tensor& seq_len,
    int64_t heads,
    double dropout_prob,
    bool enable_stream,
    bool sync,
    bool is_training) {
  TORCH_CHECK(
      input.device().is_privateuseone() && mask.device().is_privateuseone() &&
          seq_len.is_cpu(),
      "input and mask "
      "must be on MLU, seq_len must be on CPU!");
  TORCH_CHECK(
      input.is_floating_point(), "only support real floating point input!");
  TORCH_CHECK(
      input.scalar_type() == mask.scalar_type(),
      "the dtype of input and mask must "
      "be the same, but got ",
      input.scalar_type(),
      " and ",
      mask.scalar_type());
  TORCH_CHECK(
      seq_len.scalar_type() == at::ScalarType::Int ||
          seq_len.scalar_type() == at::ScalarType::Long,
      "dtype of seq_len must be Int or Long!");
  TORCH_CHECK(
      enable_stream == false,
      "currently do not support multiple streams acceleration");
  TORCH_CHECK(
      input.dim() == 1 && mask.dim() == 1 && seq_len.dim() == 1,
      "only support 1-D Tensors for input, mask and seq_len, but got ",
      input.dim(),
      "-D, ",
      mask.dim(),
      "-D and ",
      seq_len.dim(),
      "-D!");

  auto input_ = cnnl_contiguous(input, c10::MemoryFormat::Contiguous);
  auto mask_ = cnnl_contiguous(mask, c10::MemoryFormat::Contiguous);
  auto seq_len_ = seq_len.contiguous();

  int64_t cnt_i = 0, cnt_m = 0;
  for (int64_t i = 0; i < batch; ++i) {
    auto seq = seq_len_[i].item().toLong();
    auto input_slice = input_.slice(0, cnt_i, cnt_i + heads * seq * seq, 1);
    // cnnl need 4-D input
    auto input_slice_view = cnnl_view(input_slice, {1, heads, seq, seq});
    auto mask_slice = mask_.slice(0, cnt_m, cnt_m + seq, 1);
    cnnl_masked_softmax_internal(
        input_slice_view, input_slice_view, mask_slice, -1);
    cnt_i += heads * seq * seq;
    cnt_m += seq;
  }

  if (is_copy_necessary(input, input_)) {
    input.copy_(input_);
  }

  if (is_training) {
    double p1m = 1. - dropout_prob;
    return cnnl__fused_dropout(input_, p1m);
  }

  auto ret = at::native::dropout(input_, dropout_prob, is_training);
  auto ret_mask = at::Tensor();
  return std::tuple<at::Tensor, Tensor>(ret, ret_mask);
}

at::Tensor& cnnl_mask_softmax_dropout_bprop_(
    const at::Tensor& input,
    at::Tensor& grad_output,
    const at::Tensor& dropout_mask,
    int64_t batch,
    const at::Tensor& seq_len,
    int64_t heads,
    double dropout_prob,
    bool enable_stream,
    bool sync) {
  TORCH_CHECK(
      input.is_floating_point(), "only support real floating point input!");
  TORCH_CHECK(
      input.scalar_type() == grad_output.scalar_type(),
      "the dtype of input and grad_output must "
      "be the same, but got ",
      input.scalar_type(),
      " and ",
      grad_output.scalar_type());
  TORCH_CHECK(
      dropout_mask.scalar_type() == at::ScalarType::Byte,
      "dropout_mask should be torch.uint8 dtype.");
  TORCH_CHECK(
      seq_len.scalar_type() == at::ScalarType::Int ||
          seq_len.scalar_type() == at::ScalarType::Long,
      "dtype of seq_len must be Int or Long!");
  TORCH_CHECK(
      enable_stream == false,
      "currently do not support multiple streams acceleration");
  TORCH_CHECK(
      input.dim() == 1 && grad_output.dim() == 1 && dropout_mask.dim() == 1 &&
          seq_len.dim() == 1,
      "only support 1-D Tensors for input, grad_output, dropout_mask and seq_len, but got ",
      input.dim(),
      "-D, ",
      grad_output.dim(),
      "-D, ",
      dropout_mask.dim(),
      "-D and ",
      seq_len.dim(),
      "-D!");

  auto dropout_prob_ =
      c10::checked_convert<float, double>(dropout_prob, "float");

  auto input_ = cnnl_contiguous(input, c10::MemoryFormat::Contiguous);
  auto grad_output_ =
      cnnl_contiguous(grad_output, c10::MemoryFormat::Contiguous);
  auto dropout_mask_ =
      cnnl_contiguous(dropout_mask, c10::MemoryFormat::Contiguous);
  auto seq_len_ = seq_len.contiguous();

  int64_t cnt = 0;
  for (int64_t i = 0; i < batch; ++i) {
    auto seq = seq_len_[i].item().toLong();
    auto input_slice = input_.slice(0, cnt, cnt + heads * seq * seq, 1);
    // cnnl need 4-D input
    auto input_slice_view = cnnl_view(input_slice, {1, heads, seq, seq});
    auto grad_output_slice =
        grad_output_.slice(0, cnt, cnt + heads * seq * seq, 1);
    // cnnl need 4-D input
    auto grad_output_slice_view =
        cnnl_view(grad_output_slice, {1, heads, seq, seq});
    auto dropout_mask_slice =
        dropout_mask_.slice(0, cnt, cnt + heads * seq * seq, 1);
    // cnnl need 4-D input
    auto dropout_mask_slice_view =
        cnnl_view(dropout_mask_slice, {1, heads, seq, seq});
    cnnl_masked_softmax_dropout_backward_internal(
        grad_output_slice_view,
        input_slice_view,
        grad_output_slice_view,
        dropout_mask_slice_view,
        -1,
        dropout_prob_);
    cnt += heads * seq * seq;
  }

  if (is_copy_necessary(grad_output, grad_output_)) {
    grad_output.copy_(grad_output_);
  }

  return grad_output;
}

} // namespace ops
} // namespace torch_mlu
