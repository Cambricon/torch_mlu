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

#include <ATen/Dispatch.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

inline at::ScalarType get_dtype_from_self(
    const Tensor& self,
    const c10::optional<at::ScalarType>& dtype,
    bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();
  }
  at::ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return at::kLong;
  }
  return src_type;
}

at::Tensor cnnl_trace(const at::Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "expected a matrix");
  at::Tensor result;
  // Returns the ScalarType of the self tensor if the tensor is non integral
  // type
  //   // In the case, self is an integer type tensor, at::kLong is return since
  //   promote_integers
  //     // is set to true
  at::ScalarType dtype = get_dtype_from_self(self, c10::nullopt, true);
  result = at::empty({}, self.options().dtype(dtype));
  auto tmp_out = create_int_tensor_if_needed(result);
  auto self_contiguous = cast_long_to_int_if_needed(cnnl_contiguous(self));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::kComplexHalf,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "cnnl_trace",
      [&] { cnnl_trace_internal(self_contiguous, tmp_out); });
  cast_int_to_long_if_needed(tmp_out, result);
  return result;
}

} // namespace ops
} // namespace torch_mlu
