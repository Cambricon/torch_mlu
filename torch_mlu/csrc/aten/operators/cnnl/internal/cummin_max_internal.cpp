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

void cnnl_cummin_max_internal(
    at::Tensor& input,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim,
    CumType kind) {
  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_values_impl = getMluTensorImpl(values);
  auto descOutputValues = getTensorDesc(output_values_impl);
  auto output_values_ptr = mlu_data_ptr(output_values_impl);

  auto output_indices_impl = getMluTensorImpl(indices);
  auto descOutputIndices = getTensorDesc(output_indices_impl);
  auto output_indices_ptr = mlu_data_ptr(output_indices_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // malloc mlu memory
  if (kind == CumType::Cum_Min) {
    TORCH_CNNL_CHECK(cnnlCummin(
        handle,
        descInput.get(),
        input_ptr,
        dim,
        descOutputValues.get(),
        output_values_ptr,
        descOutputIndices.get(),
        output_indices_ptr));
  } else if (kind == CumType::Cum_Max) {
    TORCH_CNNL_CHECK(cnnlCummax(
        handle,
        descInput.get(),
        input_ptr,
        dim,
        descOutputValues.get(),
        output_values_ptr,
        descOutputIndices.get(),
        output_indices_ptr));
  }
}

} // namespace ops
} // namespace torch_mlu
