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

void cnnl_embedding_internal(
    const at::Tensor& weight,
    const at::Tensor& indices,
    at::Tensor& output,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  // prepare cnnl input
  auto weight_impl = getMluTensorImpl(weight);
  auto weight_ptr = mlu_data_ptr(weight_impl);
  CnnlTensorDescriptor weight_desc;
  weight_desc.set(weight, CNNL_LAYOUT_ARRAY);

  auto indices_impl = getMluTensorImpl(indices);
  auto indices_ptr = mlu_data_ptr(indices_impl);
  CnnlTensorDescriptor indices_desc;
  indices_desc.set(indices, CNNL_LAYOUT_ARRAY);

  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  CnnlTensorDescriptor output_desc;
  output_desc.set(output, CNNL_LAYOUT_ARRAY);

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlEmbeddingForward_v2(
      handle,
      weight_desc.desc(),
      weight_ptr,
      indices_desc.desc(),
      indices_ptr,
      padding_idx,
      nullptr,
      nullptr,
      output_desc.desc(),
      output_ptr));
}
} // namespace ops
} // namespace torch_mlu
