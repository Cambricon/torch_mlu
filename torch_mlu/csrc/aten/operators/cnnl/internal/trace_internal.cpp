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

void cnnl_trace_internal(const at::Tensor& self, at::Tensor& result) {
  if (self.numel() == 0) {
    result.zero_();
    return;
  }
  auto self_impl = getMluTensorImpl(self);
  auto result_impl = getMluTensorImpl(result);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_self;
  CnnlTensorDescriptor desc_result;
  desc_self.set(self);
  desc_result.set(result);
  // malloc mlu memory
  auto self_ptr = self_impl->mlu_data_ptr();
  auto result_ptr = result_impl->mlu_data_ptr();
  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetTraceWorkspaceSize(handle, desc_self.desc(), &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  TORCH_CNNL_CHECK(cnnlTrace(
      /* handle      */ handle,
      /* input_desc  */ desc_self.desc(),
      /* input       */ self_ptr,
      /* workspace   */ workspace_ptr.get(),
      /* ws_size     */ space_size,
      /* output_desc */ desc_result.desc(),
      /* output      */ result_ptr));
}

} // namespace ops
} // namespace torch_mlu
