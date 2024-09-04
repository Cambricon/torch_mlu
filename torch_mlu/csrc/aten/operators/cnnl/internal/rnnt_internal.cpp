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
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {
std::tuple<at::Tensor, std::optional<at::Tensor>> cnnl_rnnt_loss_internal(
    at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& logit_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    double clamp) {
  CnnlRNNTLossDescriptor rnntloss_desc;
  cnnlLossReduction_t reduce_mode = CNNL_LOSS_REDUCTION_NONE;

  const int64_t max_target_length = at::max(target_lengths).item().toInt();
  const int64_t max_logit_length = at::max(logit_lengths).item().toInt();
  const bool fused_log_softmax = true;
  rnntloss_desc.set(
      reduce_mode,
      blank,
      clamp,
      fused_log_softmax,
      max_logit_length,
      max_target_length);

  CnnlTensorDescriptor logits_desc, targets_desc, logit_lengths_desc,
      target_lengths_desc, costs_desc, gradients_desc;

  logits_desc.set(logits, CNNL_LAYOUT_ARRAY);
  targets_desc.set(targets, CNNL_LAYOUT_ARRAY);
  logit_lengths_desc.set(logit_lengths, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);
  target_lengths_desc.set(target_lengths, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_INT32);

  auto logits_impl = getMluTensorImpl(logits);
  auto targets_impl = getMluTensorImpl(targets);
  auto logit_lengths_impl = getMluTensorImpl(logit_lengths);
  auto target_lengths_impl = getMluTensorImpl(target_lengths);

  auto logits_ptr = mlu_data_ptr(logits_impl);
  auto targets_ptr = mlu_data_ptr(targets_impl);
  auto target_lengths_ptr = mlu_data_ptr(target_lengths_impl);
  auto logit_lengths_ptr = mlu_data_ptr(logit_lengths_impl);

  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  at::Tensor gradients = at::zeros_like(logits);

  gradients_desc.set(gradients);
  auto gradients_impl = getMluTensorImpl(gradients);
  auto gradients_ptr = mlu_data_ptr(gradients_impl);

  TORCH_CNNL_CHECK(cnnlGetRNNTLossWorkspaceSize(
      /*handle*/ handle,
      /*rnnt_loss_desc*/ rnntloss_desc.desc(),
      /*logits_desc*/ logits_desc.desc(),
      /*grads_desc*/ gradients_desc.desc(),
      &workspace_size));

  // allocate workspace
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  auto batchSize = logit_lengths.size(0);
  auto nHypos = target_lengths.size(0) / logit_lengths.size(0);

  at::Tensor costs =
      at::empty(batchSize * nHypos, logits.options().dtype(logits.dtype()));
  costs_desc.set(costs);
  auto costs_impl = getMluTensorImpl(costs);
  auto costs_ptr = mlu_data_ptr(costs_impl);

  TORCH_CNNL_CHECK(cnnlRNNTLoss(
      /*handle*/ handle,
      /*rnnt_loss_desc*/ rnntloss_desc.desc(),
      /*logits_desc*/ logits_desc.desc(),
      /*logits*/ logits_ptr,
      /*targets_desc*/ targets_desc.desc(),
      /*targets*/ targets_ptr,
      /*logit_lengths_desc*/ logit_lengths_desc.desc(),
      /*logit_lengths*/ logit_lengths_ptr,
      /*target_lengths_desc*/ target_lengths_desc.desc(),
      /*target_lengths*/ target_lengths_ptr,
      /*workspace*/ workspace_ptr.get(),
      /*workspace_size*/ workspace_size,
      /*loss_desc*/ costs_desc.desc(),
      /*loss*/ costs_ptr,
      /*grads_desc*/ gradients_desc.desc(),
      /*grads*/ gradients_ptr));

  std::optional<at::Tensor> gradients_opt = gradients;
  return std::make_tuple(costs, gradients_opt);
}

} // namespace ops
} // namespace torch_mlu
