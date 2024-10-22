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
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {
void cnnl_bce_internal(
    at::Tensor& loss,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction) {
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->mlu_data_ptr();
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  auto target_impl = getMluTensorImpl(target);
  auto target_ptr = target_impl->mlu_data_ptr();
  CnnlTensorDescriptor target_desc;
  target_desc.set(target);
  bool weight_flag = weight.defined();
  auto weight_impl = target_impl;
  auto weight_ptr = target_ptr;
  CnnlTensorDescriptor weight_desc;
  if (weight_flag) {
    weight_impl = getMluTensorImpl(weight);
    weight_ptr = weight_impl->mlu_data_ptr();
    weight_desc.set(weight);
  }

  auto output_impl = getMluTensorImpl(loss);
  auto output_ptr = output_impl->mlu_data_ptr();
  CnnlTensorDescriptor output_desc;
  output_desc.set(loss);

  size_t sz = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetBceLossWorkspaceSize(
      handle,
      self_desc.desc(),
      weight_flag ? weight_desc.desc() : nullptr,
      &sz));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(sz);

  cnnlBceLossReduction_t reduction_mode = CNNL_BCE_LOSS_MEAN;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_BCE_LOSS_NONE;
      break;
    case 1:
      reduction_mode = CNNL_BCE_LOSS_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_BCE_LOSS_SUM;
      break;
    default:
      LOG(ERROR) << "binary_cross_entropy reduciton mode is unsupported.";
      break;
  }

  TORCH_CNNL_CHECK(cnnlBceLoss(
      handle,
      self_desc.desc(),
      self_ptr,
      target_desc.desc(),
      target_ptr,
      weight_flag ? weight_desc.desc() : nullptr,
      weight_flag ? weight_ptr : nullptr,
      reduction_mode,
      ws_ptr.get(),
      sz,
      output_desc.desc(),
      output_ptr));
}

void cnnl_bce_bp_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction) {
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_input_ptr = grad_input_impl->mlu_data_ptr();
  CnnlTensorDescriptor grad_input_desc;
  grad_input_desc.set(grad_input);

  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_output_ptr = grad_output_impl->mlu_data_ptr();
  CnnlTensorDescriptor grad_output_desc;
  grad_output_desc.set(grad_output);

  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->mlu_data_ptr();
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);

  auto target_impl = getMluTensorImpl(target);
  auto target_ptr = target_impl->mlu_data_ptr();
  CnnlTensorDescriptor target_desc;
  target_desc.set(target);

  decltype(target_impl) weight_impl = nullptr;
  decltype(target_ptr) weight_ptr = nullptr;
  CnnlTensorDescriptor weight_desc;
  bool weight_flag = weight.defined();
  if (weight_flag) {
    weight_impl = getMluTensorImpl(weight);
    weight_ptr = weight_impl->mlu_data_ptr();
    weight_desc.set(weight);
  }

  cnnlBceLossReduction_t reduction_mode = CNNL_BCE_LOSS_MEAN;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_BCE_LOSS_NONE;
      break;
    case 1:
      reduction_mode = CNNL_BCE_LOSS_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_BCE_LOSS_SUM;
      break;
    default:
      LOG(ERROR) << "binary_cross_entropy reduciton mode is unsupported.";
      break;
  }

  size_t sz = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetBceLossBackwardWorkspaceSize(
      handle,
      target_desc.desc(),
      weight_flag ? weight_desc.desc() : nullptr,
      &sz));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(sz);
  TORCH_CNNL_CHECK(cnnlBceLossBackward(
      handle,
      grad_output_desc.desc(),
      grad_output_ptr,
      self_desc.desc(),
      self_ptr,
      target_desc.desc(),
      target_ptr,
      weight_flag ? weight_desc.desc() : nullptr,
      weight_flag ? weight_ptr : nullptr,
      reduction_mode,
      ws_ptr.get(),
      sz,
      grad_input_desc.desc(),
      grad_input_ptr));
}

} // namespace ops
} // namespace torch_mlu
