/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
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

#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/binaryops_util.h"
#include "aten/utils/cnnl_util.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_where_internal(
    at::Tensor& output,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto self_shape = self.sizes();
  auto other_shape = other.sizes();
  if (condition.numel() == 0) {
    return output;
  }

  // get tensor impl
  auto condition_impl = getMluTensorImpl(condition);
  auto self_impl = getMluTensorImpl(self);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  // create input desc
  CnnlTensorDescriptor descCondition;
  CnnlTensorDescriptor descSelf;
  CnnlTensorDescriptor descOther;

  auto cnnl_layout = suggest_cnnl_layout(output);
  descCondition.set(condition, cnnl_layout);
  descSelf.set(self, cnnl_layout);
  descOther.set(other, cnnl_layout);
  // create output desc
  CnnlTensorDescriptor descOutput;
  descOutput.set(output, cnnl_layout);
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSelectV2WorkspaceSize(
      handle,
      descCondition.desc(),
      descSelf.desc(),
      descOther.desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  // allocate mlu memory
  auto condition_ptr = reinterpret_cast<bool*>(mlu_data_ptr(condition_impl));
  auto self_ptr = mlu_data_ptr(self_impl);
  auto other_ptr = mlu_data_ptr(other_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  // call cnnl select api
  TORCH_CNNL_CHECK(cnnlSelectV2(
      handle,
      descCondition.desc(),
      condition_ptr,
      descSelf.desc(),
      self_ptr,
      descOther.desc(),
      other_ptr,
      workspace_ptr.get(),
      workspace_size,
      descOutput.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
