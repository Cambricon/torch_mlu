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

at::Tensor& cnnl_unfold_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  TORCH_CHECK(
      input.is_contiguous() && output.is_contiguous(),
      "Input and output tensor need be contiguous.");
  if (input.numel() == 0) {
    return output;
  }
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // int32_t parameter
  int32_t i_dimension = static_cast<int32_t>(dimension);
  int32_t i_size = static_cast<int32_t>(size);
  int32_t i_step = static_cast<int32_t>(step);

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlUnfold(
      handle,
      i_dimension,
      i_size,
      i_step,
      input_desc.get(),
      input_ptr,
      output_desc.get(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
