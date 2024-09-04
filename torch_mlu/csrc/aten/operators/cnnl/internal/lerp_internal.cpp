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
void cnnl_lerp_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight) {
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = mlu_data_ptr(self_impl);
  auto desc_self = getTensorDesc(self_impl, CNNL_LAYOUT_ARRAY);

  auto other_impl = getMluTensorImpl(end);
  auto end_ptr = mlu_data_ptr(other_impl);
  auto desc_end = getTensorDesc(other_impl, CNNL_LAYOUT_ARRAY);

  auto weight_impl = getMluTensorImpl(weight);
  auto weight_ptr = mlu_data_ptr(weight_impl);
  auto desc_weight = getTensorDesc(weight_impl, CNNL_LAYOUT_ARRAY);

  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  size_t space_size = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetLerpWorkspaceSize(
      handle,
      desc_self.get(),
      desc_end.get(),
      desc_weight.get(),
      desc_output.get(),
      &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  TORCH_CNNL_CHECK(cnnlLerp(
      handle,
      desc_self.get(),
      self_ptr,
      desc_end.get(),
      end_ptr,
      desc_weight.get(),
      weight_ptr,
      workspace_ptr.get(),
      space_size,
      desc_output.get(),
      output_ptr));
}
} // namespace ops
} // namespace torch_mlu
