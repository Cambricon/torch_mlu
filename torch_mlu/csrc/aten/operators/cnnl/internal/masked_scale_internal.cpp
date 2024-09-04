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

at::Tensor& cnnl_masked_scale_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const float scale) {
  auto input_impl = getMluTensorImpl(input);
  auto mask_impl = getMluTensorImpl(mask);
  auto output_impl = getMluTensorImpl(output);

  // get cnnl desc
  auto cnnl_suggest_layout = suggestCnnlLayout(input_impl);
  auto desc_input = getTensorDesc(input_impl, cnnl_suggest_layout);
  auto desc_mask = getTensorDesc(mask_impl, cnnl_suggest_layout);
  auto desc_output = getTensorDesc(output_impl, cnnl_suggest_layout);

  // get handle
  auto handle = getCurrentHandle();

  // masked mode
  cnnlMaskedOp_t masked_op = CNNL_MASKED_SCALE;

  // get workspace size
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(
      handle,
      masked_op,
      desc_input.get(),
      desc_mask.get(),
      nullptr,
      desc_output.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto mask_ptr = mask_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // cnnlMasked
  TORCH_CNNL_CHECK(cnnlMasked_v4(
      handle,
      masked_op,
      desc_input.get(),
      input_ptr,
      desc_mask.get(),
      mask_ptr,
      nullptr,
      nullptr,
      &scale,
      workspace_ptr.get(),
      workspace_size,
      desc_output.get(),
      output_ptr,
      nullptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
