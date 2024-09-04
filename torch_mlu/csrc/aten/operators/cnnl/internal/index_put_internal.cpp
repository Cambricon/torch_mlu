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
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_index_put_internal(
    at::Tensor& output_contiguous,
    const at::Tensor& self_contiguous,
    const std::vector<at::Tensor>& indices,
    const at::Tensor& value_contiguous,
    bool accumulate) {
  auto handle = getCurrentHandle();

  // to preserve descriptor
  c10::SmallVector<tensorDescPtr_t, 8> not_skip_desc;

  // This vector is used to preserve contiguous indices and ensure those
  // indices will not destruct until the current kernel finished running.
  // Otherwise a potential memory overlapping may occur.
  c10::SmallVector<at::Tensor, 8> not_skip_tensor;

  // To perserve transposed tensor
  c10::SmallVector<cnnlTensorDescriptor_t, 8> indices_desc;
  c10::SmallVector<void*, 8> indices_ptr_list;

  // to generate indice ptr & descriptor
  for (auto i = 0; i < indices.size(); ++i) {
    if (indices[i].defined() && indices[i].numel() != 0) {
      auto index_contiguous =
          cnnl_contiguous(indices[i], c10::MemoryFormat::Contiguous);
      not_skip_tensor.emplace_back(index_contiguous);
      auto index_impl = getMluTensorImpl(index_contiguous);
      indices_ptr_list.emplace_back(mlu_data_ptr(index_impl));

      auto index_desc = getTensorDesc(index_impl);
      indices_desc.push_back(index_desc.get());
      not_skip_desc.push_back(std::move(index_desc));
    } else {
      indices_ptr_list.push_back(nullptr);
      indices_desc.push_back(nullptr);
    }
  }
  if (not_skip_desc.size() == 0) {
    return output_contiguous;
  }

  // malloc mlu memory
  auto self_impl = getMluTensorImpl(self_contiguous);
  auto self_desc = getTensorDesc(self_impl, CNNL_LAYOUT_ARRAY);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto output_impl = getMluTensorImpl(output_contiguous);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto value_impl = getMluTensorImpl(value_contiguous);
  auto value_desc = getTensorDesc(value_impl, CNNL_LAYOUT_ARRAY);
  auto value_ptr = mlu_data_ptr(value_impl);

  // prepare cnnl workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetIndexPutWorkspaceSize(
      handle,
      self_desc.get(),
      indices_desc.data(),
      indices_desc.size(),
      value_desc.get(),
      accumulate,
      &workspace_size));
  auto temp_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  TORCH_CNNL_CHECK(cnnlIndexPut(
      handle,
      self_desc.get(),
      self_ptr,
      indices_desc.data(),
      indices_ptr_list.data(),
      indices_desc.size(),
      value_desc.get(),
      value_ptr,
      temp_ptr.get(),
      workspace_size,
      accumulate,
      true,
      output_desc.get(),
      output_ptr));
  return output_contiguous;
}

} // namespace ops
} // namespace torch_mlu
