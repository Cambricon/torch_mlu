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
    bool is_dim_none) {
  auto input_impl = getMluTensorImpl(input);
  auto desc_input = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto values_impl = getMluTensorImpl(values);
  auto desc_values = getTensorDesc(values_impl);
  auto values_ptr = mlu_data_ptr(values_impl);

  auto indices_impl = getMluTensorImpl(indices);
  auto desc_indices = getTensorDesc(indices_impl);
  auto indices_ptr = mlu_data_ptr(indices_impl);

  auto handle = getCurrentHandle();

  // int8, uint8, int16, int32, int64 are not supported
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "MLU median",
      [&] {
        TORCH_CNNL_CHECK(cnnlMedian(
            handle,
            desc_input.get(),
            input_ptr,
            desc_values.get(),
            values_ptr,
            desc_indices.get(),
            indices_ptr,
            dim,
            is_dim_none));
      });
}

} // namespace ops
} // namespace torch_mlu
