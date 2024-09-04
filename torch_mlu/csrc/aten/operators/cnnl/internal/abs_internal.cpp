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

at::Tensor& cnnl_abs_internal(at::Tensor& output, const at::Tensor& input) {
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);

  // Get Cnnl Desc
  auto descInput = getTensorDescAndCoalesceDims(input_impl);
  auto descOutput = getTensorDescAndCoalesceDims(output_impl);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto handle = getCurrentHandle();
  auto input_type = input.scalar_type();
  auto output_type = output.scalar_type();
  if (c10::isComplexType(input_type)) {
    TORCH_CHECK(
        (input_type == at::ScalarType::ComplexFloat) ||
            (input_type == at::ScalarType::ComplexDouble),
        "input must be ComplexFloat or ComplexDouble");
    TORCH_CHECK(
        (output_type == at::ScalarType::Float) ||
            (output_type == at::ScalarType::Double),
        "For complex input, output must be Float or Double");
    TORCH_CNNL_CHECK(cnnlComplexAbs(
        handle, descInput.get(), input_ptr, descOutput.get(), output_ptr));
  } else {
    TORCH_CNNL_CHECK(cnnlAbs(
        handle, descInput.get(), input_ptr, descOutput.get(), output_ptr));
  }
  return output;
}

} // namespace ops
} // namespace torch_mlu
