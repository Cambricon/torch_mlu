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
void cnnl_split_internal(
    std::vector<at::Tensor>& outputs,
    const at::Tensor& data,
    const int split_num,
    const int split_dim) {
  // get tensor desc and ptr
  auto data_impl = getMluTensorImpl(data);
  auto data_desc = getTensorDesc(data_impl);
  auto data_ptr = mlu_data_ptr(data_impl);
  c10::SmallVector<cnnlTensorDescriptor_t> out_desc_list;
  c10::SmallVector<tensorDescPtr_t> out_tensordesc_list;
  c10::SmallVector<void*> out_ptr_list;
  for (int p = 0; p < split_num; ++p) {
    auto out = outputs[p];
    auto out_impl = getMluTensorImpl(out);
    auto out_ptr = mlu_data_ptr(out_impl);
    out_tensordesc_list.emplace_back(
        getTensorDesc(out_impl, CNNL_LAYOUT_ARRAY));
    out_desc_list.emplace_back(out_tensordesc_list.back().get());
    out_ptr_list.emplace_back(out_ptr);
  }

  // get workspace
  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetSplitWorkspaceSize(handle, split_num, &workspace_size));
  at::DataPtr workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  TORCH_CNNL_CHECK(cnnlSplit(
      handle,
      split_num,
      split_dim,
      data_desc.get(),
      data_ptr,
      workspace_ptr.get(),
      workspace_size,
      out_desc_list.data(),
      out_ptr_list.data()));
  return;
}

} // namespace ops
} // namespace torch_mlu
