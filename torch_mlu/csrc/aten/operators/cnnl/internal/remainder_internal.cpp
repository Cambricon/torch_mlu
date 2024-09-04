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

void cnnl_remainder_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other) {
  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();

  // create the desc
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor other_desc;
  CnnlTensorDescriptor output_desc;
  // output tensor size/stride is re-compute in tensor iterator,
  // it will be more exacted.
  auto layout = suggest_cnnl_layout(output);
  self_desc.set(self, layout);
  other_desc.set(other, layout);
  output_desc.set(output, layout);

  // get the size of workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetFloorModWorkspaceSize(
      handle,
      self_desc.desc(),
      other_desc.desc(),
      output_desc.desc(),
      &space_size));

  // get the mlu ptr
  auto self_ptr = self_impl->mlu_data_ptr();
  auto other_ptr = other_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  // compute ops
  TORCH_CNNL_CHECK(cnnlFloorMod(
      handle,
      self_desc.desc(),
      self_ptr,
      other_desc.desc(),
      other_ptr,
      output_desc.desc(),
      output_ptr,
      workspace_ptr.get(),
      space_size));
}

} // namespace ops
} // namespace torch_mlu
