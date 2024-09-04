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

#include <algorithm>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_bitwise_op_out_internal(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other,
    const cnnlBitComputeOp_t& op_type) {
  auto suggest_self_layout = suggest_cnnl_layout(out);

  auto output_impl = getMluTensorImpl(out);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto output_desc = getTensorDesc(output_impl, suggest_self_layout);

  auto init_desc_ptr =
      [&suggest_self_layout](
          const at::Tensor& t, void*& ptr, tensorDescPtr_t& desc) {
        if (isCpuScalar(t)) {
          auto cnnl_type = getCnnlDataType(t.scalar_type());
          desc = getCpuTensorDesc(
              cnnl_type, CNNL_POINTER_MODE_HOST, suggest_self_layout);
          ptr = t.data_ptr();
        } else {
          auto impl = getMluTensorImpl(t);
          desc = getTensorDesc(impl, suggest_self_layout);
          ptr = mlu_data_ptr(impl);
        }
      };

  void* input1_ptr = nullptr;
  tensorDescPtr_t input1_desc = nullptr;
  void* input2_ptr = nullptr;
  tensorDescPtr_t input2_desc = nullptr;

  init_desc_ptr(self, input1_ptr, input1_desc);

  if (other.defined() && (op_type != CNNL_BNOT_OP)) {
    init_desc_ptr(other, input2_ptr, input2_desc);
  }

  auto handle = getCurrentHandle();

  // prepare workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetBitComputeWorkspaceSize(
      handle,
      input1_desc.get(),
      input2_desc.get(),
      output_desc.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlBitCompute_v2(
      handle,
      op_type,
      input1_desc.get(),
      input1_ptr,
      input2_desc.get(),
      input2_ptr,
      output_desc.get(),
      output_ptr,
      workspace_ptr.get(),
      workspace_size));
  return out;
}

} // namespace ops
} // namespace torch_mlu
