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

#include "aten/operators/cnnl/foreach_utils.h"

namespace torch_mlu::ops {

std::vector<at::Tensor> floating_complex_half_bfloat16(
    at::TensorList self,
    const cnnlForeachOpMode_t& mode) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(self.size());
  for (const auto& t : self) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "floating_complex_half_bfloat16",
      [&]() { cnnl_foreach_unary_op<false>(self, vec_res, mode); });
  return vec_res;
}

void floating_complex_half_bfloat16_(
    at::TensorList self,
    const cnnlForeachOpMode_t& mode) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "floating_complex_half_bfloat16_",
      [&]() { cnnl_foreach_unary_op<true>(self, self, mode); });
}

void cnnl__foreach_zero_(at::TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);

  if (!at::native::can_use_fast_route(tensors)) {
    return at::native::foreach_tensor_zero_slow_(tensors);
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "cnnl__foreach_zero_",
      [&]() {
        cnnl_foreach_unary_op<true>(tensors, tensors, CNNL_FOREACH_ZERO);
      });
}

#define FOREACH_UNARY_OP(FUNCTION, NAME, MODE)                           \
  void cnnl__foreach_##NAME##_(at::TensorList self) {                    \
    at::native::check_foreach_api_restrictions(self);                    \
    if (!at::native::can_use_fast_route(self) ||                         \
        at::native::has_integral_tensor(self, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##NAME##_slow_(self);            \
    }                                                                    \
    FUNCTION##_(self, MODE);                                             \
  }                                                                      \
                                                                         \
  std::vector<Tensor> cnnl__foreach_##NAME(at::TensorList self) {        \
    at::native::check_foreach_api_restrictions(self);                    \
    if (!at::native::can_use_fast_route(self) ||                         \
        at::native::has_integral_tensor(self, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##NAME##_slow(self);             \
    }                                                                    \
    return FUNCTION(self, MODE);                                         \
  }

FOREACH_UNARY_OP(floating_complex_half_bfloat16, sqrt, CNNL_FOREACH_SQRT);
#undef FOREACH_UNARY_OP

} // namespace torch_mlu::ops
