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

#include "ATen/native/UnaryOps.h"
#include "aten/TensorIteratorBridge.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, std::optional<at::Tensor>> cnnl_rnnt_loss(
    at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& logit_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    double clamp,
    bool fused_log_softmax) {
  TORCH_CHECK(
      logits.device().type() == targets.device().type(),
      "logits and targets must be on the same device");
  TORCH_CHECK(
      logits.device().type() == logit_lengths.device().type(),
      "logits and logit_lengths must be one the same device");
  TORCH_CHECK(
      logits.device().type() == target_lengths.device().type(),
      "logits and target_lengths must be on the same device");

  TORCH_CHECK(
      logits.scalar_type() == at::ScalarType::Float ||
          logits.scalar_type() == at::ScalarType::Half,
      "logits must be float32 or float16 (half) type");
  TORCH_CHECK(
      targets.scalar_type() == at::ScalarType::Int,
      "targets must be int32 type");
  TORCH_CHECK(
      logit_lengths.scalar_type() == at::ScalarType::Int,
      "logit_lengths must be int32 type");
  TORCH_CHECK(
      target_lengths.scalar_type() == at::ScalarType::Int,
      "target_lengths must be int32 type");

  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  TORCH_CHECK(
      logit_lengths.is_contiguous(), "logit_lengths must be contiguous");
  TORCH_CHECK(
      target_lengths.is_contiguous(), "target_lengths must be contiguous");

  TORCH_CHECK(
      logits.dim() == 4, "logits must be 4-D (batch, time, target, class)");
  TORCH_CHECK(
      targets.dim() == 2, "targets must be 2-D (batch, max target length)");
  TORCH_CHECK(logit_lengths.dim() == 1, "logit_lengths must be 1-D");
  TORCH_CHECK(target_lengths.dim() == 1, "target_lengths must be 1-D");

  TORCH_CHECK(
      logit_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and logit_lengths");
  TORCH_CHECK(
      target_lengths.size(0) == logits.size(0),
      "batch dimension mismatch between logits and target_lengths");
  TORCH_CHECK(
      targets.size(0) == logits.size(0),
      "batch dimension mismatch between logits and targets");

  TORCH_CHECK(
      blank >= 0 && blank < logits.size(-1),
      "blank must be within [0, logits.shape[-1])");

  TORCH_CHECK(
      logits.size(1) == at::max(logit_lengths).item().toInt(),
      "input length mismatch");
  TORCH_CHECK(
      logits.size(2) == at::max(target_lengths).item().toInt() + 1,
      "output length mismatch");
  TORCH_CHECK(
      targets.size(1) == at::max(target_lengths).item().toInt(),
      "target length mismatch");
  TORCH_CHECK(
      fused_log_softmax == true,
      "currently only support fused_log_softmax to be true.")
  auto result = cnnl_rnnt_loss_internal(
      logits, targets, logit_lengths, target_lengths, blank, clamp);
  return result;
}
} // namespace ops
} // namespace torch_mlu
