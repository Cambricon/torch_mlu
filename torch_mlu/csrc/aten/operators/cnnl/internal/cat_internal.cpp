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

void cnnl_cat_internal(
    std::vector<at::Tensor>& tensors,
    int64_t dim,
    c10::MemoryFormat& memory_format) {
  auto& output = tensors[0];

  // inputs
  std::vector<void*> inputs_ptr;
  std::vector<tensorDescPtr_t> inputs_desc;
  std::vector<cnnlTensorDescriptor_t> inputs_desc_t;
  auto layout = suggestCnnlLayout(memory_format);
  for (auto i = 1; i < tensors.size(); ++i) {
    auto* impl = getMluTensorImpl(tensors[i]);
    inputs_ptr.emplace_back(impl->mlu_data_ptr());
    inputs_desc.emplace_back(getTensorDesc(impl, layout));
    inputs_desc_t.emplace_back(inputs_desc.back().get());
  }

  // output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto output_desc = getTensorDesc(output_impl, layout);
  // workspace
  size_t workspace_size = 0;
  auto handle = getCurrentHandle();
  const int desc_num = inputs_desc_t.size();
  TORCH_CNNL_CHECK(
      cnnlGetConcatWorkspaceSize(handle, desc_num, &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // change dim for channels_last memory format
  int64_t axis = modify_dim_based_on_layout(dim, memory_format);

  // bf16 is nou supported
  TORCH_CNNL_CHECK(cnnlConcat(
      handle,
      desc_num,
      axis,
      inputs_desc_t.data(),
      inputs_ptr.data(),
      workspace_ptr.get(),
      workspace_size,
      output_desc.get(),
      output_ptr));
}

} // namespace ops
} // namespace torch_mlu
