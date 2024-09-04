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

#include "ATen/NativeFunctions.h"

#include "aten/operators/cnnl/internal/cnnl_internal.h"
namespace torch_mlu {
namespace ops {
at::Tensor& cnnl_nan_to_num_internal(
    at::Tensor& output,
    const at::Tensor& input,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  descInput.set(input, CNNL_LAYOUT_ARRAY);
  descOutput.set(output, CNNL_LAYOUT_ARRAY);

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "nan_to_num_internal",
      [&]() {
        auto nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
        auto pos_inf_replacement = pos_inf.has_value()
            ? static_cast<scalar_t>(pos_inf.value())
            : std::numeric_limits<scalar_t>::max();
        auto neg_inf_replacement = neg_inf.has_value()
            ? static_cast<scalar_t>(neg_inf.value())
            : std::numeric_limits<scalar_t>::lowest();
        TORCH_CNNL_CHECK(cnnlNanToNum(
            handle,
            descInput.desc(),
            input_ptr,
            nan_replacement,
            pos_inf_replacement,
            neg_inf_replacement,
            descOutput.desc(),
            output_ptr));
      });
  return output;
}

} // namespace ops
} // namespace torch_mlu
