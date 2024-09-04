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
#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_index_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const std::vector<at::Tensor>& indices) {
  // To initialize indices ptr with nullptr (for dim check in cnnl).
  // TODO(CNNLCORE-13367): CNNL kernel has a weird check for this.
  c10::SmallVector<void*, CNNL_MAX_DIM_SIZE> indices_ptr(CNNL_MAX_DIM_SIZE);
  c10::SmallVector<tensorDescPtr_t, CNNL_MAX_DIM_SIZE> desc_pool;
  desc_pool.resize(CNNL_MAX_DIM_SIZE);

  // To initialize cnnlTensorDescriptor_t with nullptr (for dim check in cnnl).
  c10::SmallVector<cnnlTensorDescriptor_t, CNNL_MAX_DIM_SIZE> indices_desc(
      CNNL_MAX_DIM_SIZE);

  size_t scalar_indice_num = 0;
  for (int i = 0; i < indices.size(); ++i) {
    if (indices[i].defined()) {
      if (indices[i].dim() == 0)
        scalar_indice_num++;
      auto impl = getMluTensorImpl(indices[i]);
      desc_pool[i] = getTensorDesc(impl);
      indices_desc[i] = desc_pool[i].get();
      indices_ptr[i] = mlu_data_ptr(impl);
    } else {
      indices_ptr[i] = nullptr;
      indices_desc[i] = nullptr;
    }
  }

  // if all indices is scalar and indices.size() equal to self.dim()
  // out will be scalar
  bool out_to_scalar = false;
  if (indices.size() == self.dim() && indices.size() == scalar_indice_num) {
    out_to_scalar = true;
  }
  // Self tensor
  auto self_impl = getMluTensorImpl(self);
  auto self_desc = getTensorDesc(self_impl);
  auto self_ptr = mlu_data_ptr(self_impl);

  // Get output sizes.
  auto handle = getCurrentHandle();
  int output_dim = 0;
  c10::SmallVector<int64_t, CNNL_MAX_DIM_SIZE> output_sizes(CNNL_MAX_DIM_SIZE);
  if (!out_to_scalar) {
    TORCH_CNNL_CHECK(cnnlGetAdvancedIndexOutputDim_v2(
        handle,
        self_desc.get(),
        indices_desc.data(),
        &output_dim,
        output_sizes.data()));
  }
  output_sizes.resize(output_dim);
  cnnl_resize_(output, output_sizes, c10::MemoryFormat::Contiguous);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // prepare cnnl workspace
  // For Bool AdavancedIndex, the workspace will be zero.
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetAdvancedIndexWorkspaceSize(
      handle, self_desc.get(), indices_desc.data(), &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // call cnnl AdvancedIndex v2 interface.
  TORCH_CNNL_CHECK(cnnlAdvancedIndex_v2(
      handle,
      self_desc.get(),
      self_ptr,
      indices_desc.data(),
      indices_ptr.data(),
      workspace_ptr.get(),
      workspace_size,
      output_desc.get(),
      output_ptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr));

  return output;
}

} // namespace ops
} // namespace torch_mlu
