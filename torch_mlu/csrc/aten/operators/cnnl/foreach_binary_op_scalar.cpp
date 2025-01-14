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
#include "ATen/native/BinaryOps.h"

namespace torch_mlu::ops {

std::vector<at::Tensor> all_types_complex_bool_half_bfloat16(
    at::TensorList tensors1,
    const at::Scalar& scalar,
    const cnnlForeachOpMode_t& mode) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "all_types_complex_bool_half_bfloat16",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_binary_tensors_op<opmath_t, false>(
            tensors1,
            {},
            vec_res,
            {},
            scalar.to<opmath_t>(),
            at::Tensor(),
            1.0,
            mode,
            cnnlForeachBinaryMode_t::FOREACH_BINARY_SCALAR);
      });
  return vec_res;
}

void all_types_complex_bool_half_bfloat16_(
    at::TensorList tensors1,
    const at::Scalar& scalar,
    const cnnlForeachOpMode_t& mode) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "all_types_complex_bool_half_bfloat16",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_binary_tensors_op<opmath_t, true>(
            tensors1,
            {},
            tensors1,
            {},
            scalar.to<opmath_t>(),
            at::Tensor(),
            1.0,
            mode,
            cnnlForeachBinaryMode_t::FOREACH_BINARY_SCALAR);
      });
  increment_version(tensors1);
}

#define FOREACH_BINARY_OP_SCALAR(FUNCTION, NAME, MODE, DIVISION_OP)      \
  void cnnl__foreach_##NAME##_(                                          \
      at::TensorList tensors, const at::Scalar& scalar) {                \
    at::native::check_foreach_api_restrictions(tensors);                 \
    if (MODE == CNNL_FOREACH_SUB) {                                      \
      at::native::sub_check(tensors[0], scalar);                         \
    }                                                                    \
    if (!at::native::can_use_fast_route(tensors, scalar, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow_(    \
          tensors, scalar);                                              \
    }                                                                    \
    FUNCTION##_(tensors, scalar, MODE);                                  \
  }                                                                      \
                                                                         \
  std::vector<Tensor> cnnl__foreach_##NAME(                              \
      at::TensorList tensors, const at::Scalar& scalar) {                \
    at::native::check_foreach_api_restrictions(tensors);                 \
    if (MODE == CNNL_FOREACH_SUB) {                                      \
      at::native::sub_check(tensors[0], scalar);                         \
    }                                                                    \
    if (!at::native::can_use_fast_route(tensors, scalar, DIVISION_OP)) { \
      return at::native::foreach_tensor_##NAME##_scalar_kernel_slow(     \
          tensors, scalar);                                              \
    }                                                                    \
    return FUNCTION(tensors, scalar, MODE);                              \
  }

FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    add,
    CNNL_FOREACH_ADD,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    div,
    CNNL_FOREACH_DIV,
    /*div_op*/ true);
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    mul,
    CNNL_FOREACH_MUL,
    /*div_op*/ false);
FOREACH_BINARY_OP_SCALAR(
    all_types_complex_bool_half_bfloat16,
    sub,
    CNNL_FOREACH_SUB,
    /*div_op*/ false);

#undef FOREACH_BINARY_OP_SCALAR

} // namespace torch_mlu::ops
