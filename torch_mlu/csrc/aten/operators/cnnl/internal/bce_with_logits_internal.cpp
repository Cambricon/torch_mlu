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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_bce_with_logits_internal(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& pos_weight,
    int64_t reduction,
    at::Tensor& output) {
  TORCH_CHECK(
      target.scalar_type() == self.scalar_type(),
      "Expected same dtype of self and target, but got dtype ",
      self.dtype(),
      " for argument 'self' and ",
      target.dtype(),
      " for argument 'target'");
  std::vector<int64_t> output_size;
  auto self_impl = getMluTensorImpl(self);
  auto target_impl = getMluTensorImpl(target);
  bool weight_flag = weight.defined();
  bool pos_weight_flag = pos_weight.defined();
  cnnlBceWithLogitsReduction_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_BCE_WITH_LOGITS_NONE;
      output_size = self.sizes().vec();
      break;
    case 1:
      reduction_mode = CNNL_BCE_WITH_LOGITS_MEAN;
      output_size = {};
      break;
    case 2:
      reduction_mode = CNNL_BCE_WITH_LOGITS_SUM;
      output_size = {};
      break;
    default:
      LOG(ERROR) << "bce_with_logits reduction mode is unavailable";
      break;
  }

  if (output_size.empty()) {
    output =
        at::empty(output_size, self.options(), c10::MemoryFormat::Contiguous);
  } else {
    output =
        at::empty(output_size, self.options(), self.suggest_memory_format());
  }
  auto output_impl = getMluTensorImpl(output);

  auto self_desc = getTensorDesc(self_impl);
  auto target_desc = getTensorDesc(target_impl);
  auto output_desc = getTensorDesc(output_impl);
  tensorDescPtr_t weight_desc;
  tensorDescPtr_t pos_weight_desc;
  auto self_ptr = mlu_data_ptr(self_impl);
  auto target_ptr = mlu_data_ptr(target_impl);
  void* weight_ptr = nullptr;
  void* pos_weight_ptr = nullptr;
  if (weight_flag) {
    auto weight_impl = getMluTensorImpl(weight);
    weight_ptr = mlu_data_ptr(weight_impl);
    weight_desc = getTensorDesc(weight_impl);
  }
  if (pos_weight_flag) {
    auto pos_weight_impl = getMluTensorImpl(pos_weight);
    pos_weight_ptr = mlu_data_ptr(pos_weight_impl);
    pos_weight_desc = getTensorDesc(pos_weight_impl);
  }
  auto output_ptr = mlu_data_ptr(output_impl);

  size_t sz = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetBceWithLogitsWorkspaceSize(
      handle,
      self_desc.get(),
      weight_flag ? weight_desc.get() : nullptr,
      pos_weight_flag ? pos_weight_desc.get() : nullptr,
      &sz));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(sz);
  cnnlComputationPreference_t mode = CNNL_COMPUTATION_FAST;

  TORCH_CNNL_CHECK(cnnlBceWithLogits_v2(
      handle,
      mode,
      self_desc.get(),
      self_ptr,
      target_desc.get(),
      target_ptr,
      weight_flag ? weight_desc.get() : nullptr,
      weight_ptr,
      pos_weight_flag ? pos_weight_desc.get() : nullptr,
      pos_weight_ptr,
      reduction_mode,
      ws_ptr.get(),
      sz,
      output_desc.get(),
      output_ptr));
}

} // namespace ops
} // namespace torch_mlu
