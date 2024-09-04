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

// This helper class does 2 jobs:
// 1. Wrap and accelerate CNNL input descs ctors/dtors
// 2. (TODO) Parallelize CNNL descs ctors/dtors after CNNLCORE-18726

class ConcatDescMgr {
 public:
  ConcatDescMgr() = default;

  ~ConcatDescMgr() {
    for (auto desc : raw_descs_) {
      cnnlDestroyTensorDescriptor(desc);
    }
  }

  void build(
      const at::MaterializedITensorListRef& materialized,
      const size_t num_of_inputs,
      const cnnlTensorLayout_t layout) {
    ptrs_.resize(num_of_inputs);
    raw_descs_.resize(num_of_inputs);

    for (const auto i : c10::irange(num_of_inputs)) {
      const at::Tensor& t = materialized[i];
      auto* impl = getMluTensorImpl(t);
      ptrs_[i] = mlu_data_ptr(impl);
      auto desc_unique_ptr = getTensorDesc(impl, layout);
      raw_descs_[i] = desc_unique_ptr.release();
    }
  }

  c10::SmallVector<void*, 16> ptrs_;
  c10::SmallVector<cnnlTensorDescriptor_t, 16> raw_descs_;
};

void cnnl_cat_internal(
    const at::ITensorListRef& inputs,
    const at::Tensor& output,
    const int64_t dim,
    const c10::MemoryFormat memory_format) {
  auto materialized = inputs.materialize();
  auto num_of_inputs = materialized.size();
  auto layout = suggestCnnlLayout(memory_format);

  // input
  ConcatDescMgr desc_mgr;
  desc_mgr.build(materialized, num_of_inputs, layout);

  // output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto output_desc = getTensorDesc(output_impl, layout);

  // workspace
  size_t workspace_size = 0;
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(
      cnnlGetConcatWorkspaceSize(handle, num_of_inputs, &workspace_size));
  void* workspace_ptr = workspace_size
      ? torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size).get()
      : nullptr;

  // change dim for channels_last memory format
  int64_t axis = modify_dim_based_on_layout(dim, memory_format);

  // bf16 is nou supported
  TORCH_CNNL_CHECK(cnnlConcat(
      handle,
      num_of_inputs,
      axis,
      desc_mgr.raw_descs_.data(),
      desc_mgr.ptrs_.data(),
      workspace_ptr,
      workspace_size,
      output_desc.get(),
      output_ptr));
}

} // namespace ops
} // namespace torch_mlu
