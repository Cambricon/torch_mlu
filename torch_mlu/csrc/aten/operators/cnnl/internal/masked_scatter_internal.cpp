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
  auto mask_impl = getMluTensorImpl(mask);
  auto source_impl = getMluTensorImpl(source);
  auto output_impl = getMluTensorImpl(output);
  // set descriptor config
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_mask;
  CnnlTensorDescriptor desc_source;
  CnnlTensorDescriptor desc_output;

  // cnnlMasked_v4 only supports CNNL_LAYOUT_ARRAY layout
  auto layout = CNNL_LAYOUT_ARRAY;
  desc_input.set(input, layout);
  desc_mask.set(mask, layout);
  desc_source.set(source, layout);
  desc_output.set(output, layout);

  // get handle
  auto handle = getCurrentHandle();

  // masked mode
  cnnlMaskedOp_t masked_op = CNNL_MASKED_SCATTER;

  // get workspace size
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(
      handle,
      masked_op,
      desc_input.desc(),
      desc_mask.desc(),
      desc_source.desc(),
      desc_output.desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto mask_ptr = mask_impl->mlu_data_ptr();
  auto source_ptr = source_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // cnnlMasked
  TORCH_CNNL_CHECK(cnnlMasked_v4(
      handle,
      masked_op,
      desc_input.desc(),
      input_ptr,
      desc_mask.desc(),
      mask_ptr,
      desc_source.desc(),
      source_ptr,
      0,
      workspace_ptr.get(),
      workspace_size,
      desc_output.desc(),
      output_ptr,
      nullptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
