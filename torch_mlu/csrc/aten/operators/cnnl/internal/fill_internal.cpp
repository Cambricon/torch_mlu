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
#include "aten/utils/dispatch.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_fill_internal(at::Tensor& input, const at::Scalar& other) {
  auto input_impl = getMluTensorImpl(input);
  auto handle = getCurrentHandle();
  auto input_ptr = mlu_data_ptr(input_impl);
  auto descInput = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::kBool,
      at::kBFloat16,
      at::kHalf,
      input.scalar_type(),
      "fill_internal_with_scalar",
      [&] {
        using torch_mlu_scalar_t = torch_mlu::Convert64BitTo32Bit_t<scalar_t>;
        torch_mlu_scalar_t scalar_value = other.to<torch_mlu_scalar_t>();
        TORCH_CNNL_CHECK(cnnlFill_v3(
            handle,
            CNNL_POINTER_MODE_HOST,
            (void*)(&scalar_value),
            descInput.get(),
            input_ptr));
      });
  return input;
}

at::Tensor& cnnl_fill_internal(at::Tensor& input, const at::Tensor& value) {
  if (isCpuScalar(value)) {
    return cnnl_fill_internal(input, value.item());
  }
  auto input_impl = getMluTensorImpl(input);
  auto handle = getCurrentHandle();
  auto descInput = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  void* device_value_ptr = mlu_data_ptr(getMluTensorImpl(value));
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::kBool,
      at::kBFloat16,
      at::kHalf,
      input.scalar_type(),
      "fill_internal_with_tensor",
      [&] {
        TORCH_CNNL_CHECK(cnnlFill_v3(
            handle,
            CNNL_POINTER_MODE_DEVICE,
            device_value_ptr,
            descInput.get(),
            input_ptr));
      });
  return input;
}

} // namespace ops
} // namespace torch_mlu
