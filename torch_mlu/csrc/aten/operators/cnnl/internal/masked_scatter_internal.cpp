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

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_masked_scatter_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const at::Tensor& source) {
  auto input_impl = getMluTensorImpl(input);
  auto desc_input = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto mask_impl = getMluTensorImpl(mask);
  auto desc_mask = getTensorDesc(mask_impl, CNNL_LAYOUT_ARRAY);
  auto mask_ptr = mlu_data_ptr(mask_impl);

  auto source_impl = getMluTensorImpl(source);
  auto desc_source = getTensorDesc(source_impl, CNNL_LAYOUT_ARRAY);
  auto source_ptr = mlu_data_ptr(source_impl);

  auto output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get handle
  auto handle = getCurrentHandle();

  // masked mode
  cnnlMaskedOp_t masked_op = CNNL_MASKED_SCATTER;

  // get workspace size
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(
      handle,
      masked_op,
      desc_input.get(),
      desc_mask.get(),
      desc_source.get(),
      desc_output.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // cnnlMasked
  TORCH_CNNL_CHECK(cnnlMasked_v4(
      handle,
      masked_op,
      desc_input.get(),
      input_ptr,
      desc_mask.get(),
      mask_ptr,
      desc_source.get(),
      source_ptr,
      0,
      workspace_ptr.get(),
      workspace_size,
      desc_output.get(),
      output_ptr,
      nullptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
