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

#include "cnnl_internal.h"
#include "aten/operators/cnnl/internal/foreach_common_utils.h"

namespace torch_mlu::ops {

void cnnl_foreach_norm_internal(
    at::TensorList tensors,
    at::TensorList outputs,
    const float pnorm) {
  auto handle = getCurrentHandle();
  ForeachOPTensorScalarHandle<1, 1, false, true> tensor_desc_ptr(
      {tensors, outputs}, {});
  const int64_t tensor_num = tensor_desc_ptr.get_tensor_num();
  if (tensor_num == 0)
    return;
  auto [input_desc_array, input_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<0>();
  auto [output_desc_array, output_ptr_array] =
      tensor_desc_ptr.template get_output_tensor_desc_and_ptr<0>();

  size_t workspace_size = 0;
  at::DataPtr workspace_ptr;
  TORCH_CNNL_CHECK(cnnlGetForeachNormWorkspaceSize(
      handle, tensor_num, input_desc_array, &workspace_size));
  if (workspace_size != 0) {
    workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  }

  TORCH_CNNL_CHECK(cnnlForeachNorm(
      handle,
      tensor_num,
      input_desc_array,
      input_ptr_array,
      &pnorm,
      workspace_ptr.get(),
      workspace_size,
      output_desc_array,
      output_ptr_array));
}
} // namespace torch_mlu::ops
