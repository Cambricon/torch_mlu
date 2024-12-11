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

void cnnl_median_internal(
    const at::Tensor& input,
    int64_t dim,
    at::Tensor& values,
    at::Tensor& indices,
    bool is_dim_none,
    bool ignore_nan) {
  auto input_impl = getMluTensorImpl(input);
  auto values_impl = getMluTensorImpl(values);
  auto indices_impl = getMluTensorImpl(indices);

  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_values;
  CnnlTensorDescriptor desc_indices;

  desc_input.set(input);
  desc_values.set(values);
  desc_indices.set(indices);

  auto input_ptr = input_impl->mlu_data_ptr();
  auto values_ptr = values_impl->mlu_data_ptr();
  auto indices_ptr = indices_impl->mlu_data_ptr();

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetMedianWorkspaceSize(
      handle,
      dim,
      is_dim_none,
      ignore_nan,
      desc_input.desc(),
      desc_values.desc(),
      desc_indices.desc(),
      &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // int8, uint8, int16, int32, int64 are not supported
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "MLU median",
      [&] {
        TORCH_CNNL_CHECK(cnnlMedian_v2(
            handle,
            dim,
            is_dim_none,
            ignore_nan,
            desc_input.desc(),
            input_ptr,
            workspace_ptr.get(),
            workspace_size,
            desc_values.desc(),
            values_ptr,
            desc_indices.desc(),
            indices_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
