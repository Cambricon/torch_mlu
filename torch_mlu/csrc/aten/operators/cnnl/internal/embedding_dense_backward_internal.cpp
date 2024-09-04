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
void cnnl_embedding_dense_backward_internal(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    at::Tensor& output) {
  // handle scalar tensor.
  auto indices_dim = indices.dim();
  auto grad_new = grad_output;
  if (indices_dim == 0 && indices.numel() == 1) {
    grad_new = at::unsqueeze(grad_output, 0);
  }

  auto grad_impl = getMluTensorImpl(grad_new);
  auto grad_ptr = mlu_data_ptr(grad_impl);

  auto indices_impl = getMluTensorImpl(indices);
  auto indices_ptr = mlu_data_ptr(indices_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto layout = suggestCnnlLayout(grad_impl);
  auto grad_desc = getTensorDesc(grad_impl, layout);
  auto indices_desc = getTensorDesc(indices_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);

  auto handle = getCurrentHandle();
  // get workspace size
  size_t tmp_size = 0;
  TORCH_CNNL_CHECK(cnnlGetEmbeddingBackwardWorkspaceSize(
      handle,
      grad_desc.get(),
      output_desc.get(),
      scale_grad_by_freq,
      &tmp_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(tmp_size);
  TORCH_CNNL_CHECK(cnnlEmbeddingBackward(
      handle,
      padding_idx,
      scale_grad_by_freq,
      indices_desc.get(),
      indices_ptr,
      grad_desc.get(),
      grad_ptr,
      workspace_ptr.get(),
      tmp_size,
      output_desc.get(),
      output_ptr));
}

} // namespace ops
} // namespace torch_mlu
