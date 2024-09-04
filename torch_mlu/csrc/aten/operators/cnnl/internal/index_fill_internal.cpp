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
#include "c10/core/ScalarType.h"

namespace torch_mlu {
namespace ops {

void cnnl_index_fill_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Scalar value) {
  auto input_impl = getMluTensorImpl(self);
  auto input_ptr = mlu_data_ptr(input_impl);
  CnnlTensorDescriptor descInput;
  descInput.set(self);

  auto index_impl = getMluTensorImpl(index);
  auto index_ptr = mlu_data_ptr(index_impl);
  CnnlTensorDescriptor descIndex;
  descIndex.set(index);

  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  CnnlTensorDescriptor descOutput;
  descOutput.set(output);

  auto handle = getCurrentHandle();
  const cnnlPointerMode_t pointer_mode = CNNL_POINTER_MODE_HOST;

  AT_DISPATCH_ALL_TYPES_AND3(
      at::kBool,
      at::kHalf,
      at::kBFloat16,
      self.scalar_type(),
      "index_fill_internal_with_scalar",
      [&] {
        auto cnnl_value =
            value.to<torch_mlu::Convert64BitTo32Bit_t<scalar_t>>();
        TORCH_CNNL_CHECK(cnnlIndexFill_v2(
            handle,
            dim,
            pointer_mode,
            &cnnl_value,
            descInput.desc(),
            input_ptr,
            descIndex.desc(),
            index_ptr,
            descOutput.desc(),
            output_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
