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

std::vector<at::Tensor> all_floating_types_with_half_bfloat16(
    at::TensorList tensors1,
    at::TensorList tensors2,
    const cnnlForeachOpMode_t& mode,
    const at::Scalar& alpha = 1) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "all_floating_types_with_half_bfloat16",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_binary_tensors_op<opmath_t, false>(
            tensors1,
            tensors2,
            vec_res,
            {},
            1.0,
            at::Tensor(),
            alpha.to<opmath_t>(),
            mode,
            cnnlForeachBinaryMode_t::FOREACH_BINARY_TENSOR_LIST);
      });
  return vec_res;
}

void all_floating_types_with_half_bfloat16_(
    at::TensorList tensors1,
    at::TensorList tensors2,
    const cnnlForeachOpMode_t& mode,
    const at::Scalar& alpha = 1) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "all_floating_types_with_half_bfloat16",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_binary_tensors_op<opmath_t, true>(
            tensors1,
            tensors2,
            tensors1,
            {},
            1.0,
            at::Tensor(),
            alpha.to<opmath_t>(),
            mode,
            cnnlForeachBinaryMode_t::FOREACH_BINARY_TENSOR_LIST);
      });
  increment_version(tensors1);
}

#define FOREACH_BINARY_OP_LIST_ALPHA(FUNCTION, NAME, MODE)              \
  void cnnl__foreach_##NAME##_(                                         \
      at::TensorList tensors1,                                          \
      at::TensorList tensors2,                                          \
      const at::Scalar& alpha) {                                        \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);     \
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(     \
          tensors1, tensors2, alpha);                                   \
    }                                                                   \
    FUNCTION##_(tensors1, tensors2, MODE, alpha);                       \
  }                                                                     \
                                                                        \
  std::vector<Tensor> cnnl__foreach_##NAME(                             \
      at::TensorList tensors1,                                          \
      at::TensorList tensors2,                                          \
      const at::Scalar& alpha) {                                        \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);     \
    if (!at::native::can_use_fast_route({tensors1, tensors2}, alpha)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(      \
          tensors1, tensors2, alpha);                                   \
    }                                                                   \
    return FUNCTION(tensors1, tensors2, MODE, alpha);                   \
  }

#define FOREACH_BINARY_OP_LIST(FUNCTION, NAME, MODE, DIVISION_OP)           \
  void cnnl__foreach_##NAME##_(                                             \
      at::TensorList tensors1, at::TensorList tensors2) {                   \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);         \
    if (!at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow_(         \
          tensors1, tensors2);                                              \
    }                                                                       \
                                                                            \
    FUNCTION##_(tensors1, tensors2, MODE);                                  \
  }                                                                         \
                                                                            \
  std::vector<Tensor> cnnl__foreach_##NAME(                                 \
      at::TensorList tensors1, at::TensorList tensors2) {                   \
    at::native::check_foreach_api_restrictions(tensors1, tensors2);         \
    if (!at::native::can_use_fast_route(tensors1, tensors2, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_list_kernel_slow(          \
          tensors1, tensors2);                                              \
    }                                                                       \
                                                                            \
    return FUNCTION(tensors1, tensors2, MODE);                              \
  }

// add and sub pytorch support all_types_complex_bool_half_bfloat16, but only
// support half, bfloat16, float in mlu side, scalar list support half,
// bfloat16, float, int32.
FOREACH_BINARY_OP_LIST_ALPHA(
    all_floating_types_with_half_bfloat16,
    add,
    CNNL_FOREACH_ADD);
FOREACH_BINARY_OP_LIST(
    all_floating_types_with_half_bfloat16,
    mul,
    CNNL_FOREACH_MUL,
    /*division_op*/ false);
FOREACH_BINARY_OP_LIST_ALPHA(
    all_floating_types_with_half_bfloat16,
    sub,
    CNNL_FOREACH_SUB);

#undef FOREACH_BINARY_OP_LIST
#undef FOREACH_BINARY_OP_LIST_ALPHA

} // namespace torch_mlu::ops
