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
void cnnl_fmod_internal(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto cnnl_layout = suggest_cnnl_layout(self);

  auto result_impl = getMluTensorImpl(result);
  auto result_ptr = mlu_data_ptr(result_impl);
  CnnlTensorDescriptor result_desc;
  result_desc.set(result, cnnl_layout);

  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = mlu_data_ptr(self_impl);
  CnnlTensorDescriptor self_desc;
  self_desc.set(self, cnnl_layout);

  auto other_impl = getMluTensorImpl(other);
  auto other_ptr = mlu_data_ptr(other_impl);
  CnnlTensorDescriptor other_desc;
  other_desc.set(other, cnnl_layout);

  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetFloorModTruncWorkspaceSize(
      handle,
      self_desc.desc(),
      other_desc.desc(),
      result_desc.desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlFloorModTrunc(
      handle,
      self_desc.desc(),
      self_ptr,
      other_desc.desc(),
      other_ptr,
      result_desc.desc(),
      result_ptr,
      workspace_ptr.get(),
      workspace_size));
}

} // namespace ops
} // namespace torch_mlu
