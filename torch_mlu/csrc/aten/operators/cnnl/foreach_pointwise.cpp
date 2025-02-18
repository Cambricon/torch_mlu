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
#include "ATen/native/PointwiseOps.h"

namespace torch_mlu::ops {

// scalar mode
std::vector<at::Tensor> foreach_pointwise_scalar(
    at::TensorList input,
    at::TensorList tensors1,
    at::TensorList tensors2,
    const at::Scalar& scalar,
    const cnnlForeachOpMode_t& op_mode) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_scalar_mlu",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_pointwise_op<opmath_t, false>(
            input,
            tensors1,
            tensors2,
            vec_res,
            scalar.to<opmath_t>(),
            {},
            op_mode,
            cnnlForeachPointWiseMode_t::FOREACH_POINTWISE_SCALAR);
      });
  return vec_res;
}

// scalar in-place mode
void foreach_pointwise_scalar_(
    at::TensorList input,
    at::TensorList tensors1,
    at::TensorList tensors2,
    const at::Scalar& scalar,
    const cnnlForeachOpMode_t& op_mode) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_scalar_mlu_",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_pointwise_op<opmath_t, true>(
            input,
            tensors1,
            tensors2,
            input,
            scalar.to<opmath_t>(),
            {},
            op_mode,
            cnnlForeachPointWiseMode_t::FOREACH_POINTWISE_SCALAR);
      });
  increment_version(input);
  for (const auto& t : input) {
    TORCH_CHECK(t.defined(), "not defined")
  }
}

// scalar list mode
std::vector<at::Tensor> foreach_pointwise_scalarlist(
    at::TensorList input,
    at::TensorList tensors1,
    at::TensorList tensors2,
    at::ArrayRef<at::Scalar> scalars,
    const cnnlForeachOpMode_t& op_mode) {
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_scalar_list_mlu",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_pointwise_op<opmath_t, false>(
            input,
            tensors1,
            tensors2,
            vec_res,
            1.0,
            scalars,
            op_mode,
            cnnlForeachPointWiseMode_t::FOREACH_POINTWISE_SCALAR_LIST);
      });
  return vec_res;
}

// scalar list in-place mode
void foreach_pointwise_scalarlist_(
    at::TensorList input,
    at::TensorList tensors1,
    at::TensorList tensors2,
    at::ArrayRef<at::Scalar> scalars,
    const cnnlForeachOpMode_t& op_mode) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_scalar_list_mlu_",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_pointwise_op<opmath_t, true>(
            input,
            tensors1,
            tensors2,
            input,
            1.0,
            scalars,
            op_mode,
            cnnlForeachPointWiseMode_t::FOREACH_POINTWISE_SCALAR_LIST);
      });
  increment_version(input);
}

#define FOREACH_POINTWISE_OP_SCALAR(NAME, MODE, FUNC)                      \
  /* foreach pointwise scalar mode */                                      \
  std::vector<at::Tensor> cnnl__foreach_##NAME(                            \
      at::TensorList input,                                                \
      at::TensorList tensors1,                                             \
      at::TensorList tensors2,                                             \
      const at::Scalar& scalar) {                                          \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2); \
    if (!at::native::can_use_fast_route(                                   \
            {input, tensors1, tensors2}, scalar) ||                        \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {  \
      return at::native::foreach_tensor_##NAME##_scalar_slow(              \
          input, tensors1, tensors2, scalar);                              \
    }                                                                      \
                                                                           \
    return FUNC(input, tensors1, tensors2, scalar, MODE);                  \
  }                                                                        \
                                                                           \
  /* foreach pointwise scalar in-place mode */                             \
  void cnnl__foreach_##NAME##_(                                            \
      at::TensorList input,                                                \
      at::TensorList tensors1,                                             \
      at::TensorList tensors2,                                             \
      const at::Scalar& scalar) {                                          \
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2); \
    if (!at::native::can_use_fast_route(                                   \
            {input, tensors1, tensors2}, scalar) ||                        \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {  \
      return at::native::foreach_tensor_##NAME##_scalar_slow_(             \
          input, tensors1, tensors2, scalar);                              \
    }                                                                      \
                                                                           \
    FUNC##_(input, tensors1, tensors2, scalar, MODE);                      \
    for (const auto& t : input) {                                          \
      TORCH_CHECK(t.defined(), "not defined")                              \
    }                                                                      \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST(NAME, MODE, FUNC)                 \
  /* foreach pointwise scalar list mode */                                \
  std::vector<at::Tensor> cnnl__foreach_##NAME(                           \
      at::TensorList input,                                               \
      at::TensorList tensors1,                                            \
      at::TensorList tensors2,                                            \
      at::ArrayRef<at::Scalar> scalars) {                                 \
    at::native::check_foreach_api_restrictions(                           \
        input, tensors1, tensors2, scalars);                              \
    if (!at::native::can_use_fast_route(                                  \
            {input, tensors1, tensors2}, scalars) ||                      \
        at::native::has_integral_tensor(input, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(         \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    return FUNC(input, tensors1, tensors2, scalars, MODE);                \
  }                                                                       \
                                                                          \
  /* foreach pointwise scalar list in-place mode */                       \
  void cnnl__foreach_##NAME##_(                                           \
      at::TensorList input,                                               \
      at::TensorList tensors1,                                            \
      at::TensorList tensors2,                                            \
      at::ArrayRef<at::Scalar> scalars) {                                 \
    at::native::check_foreach_api_restrictions(                           \
        input, tensors1, tensors2, scalars);                              \
    if (!at::native::can_use_fast_route(                                  \
            {input, tensors1, tensors2}, scalars) ||                      \
        at::native::has_integral_tensor(input, /* includeBool */ true)) { \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(        \
          input, tensors1, tensors2, scalars);                            \
    }                                                                     \
                                                                          \
    FUNC##_(input, tensors1, tensors2, scalars, MODE);                    \
  }

#define FOREACH_POINTWISE_OP_TENSOR(NAME, MODE, FUNC)                      \
  /* foreach pointwise tensor mode */                                      \
  std::vector<at::Tensor> cnnl__foreach_##NAME(                            \
      at::TensorList input,                                                \
      at::TensorList tensors1,                                             \
      at::TensorList tensors2,                                             \
      const Tensor& scalars_) {                                            \
    auto scalars =                                                         \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size()); \
    at::native::check_foreach_api_restrictions(                            \
        input, tensors1, tensors2, scalars);                               \
    if (!at::native::can_use_fast_route({input, tensors1, tensors2}) ||    \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {  \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow(          \
          input, tensors1, tensors2, scalars);                             \
    }                                                                      \
                                                                           \
    return FUNC(input, tensors1, tensors2, scalars, MODE);                 \
  }                                                                        \
                                                                           \
  /* foreach pointwise scalar tensor in-place mode */                      \
  void cnnl__foreach_##NAME##_(                                            \
      at::TensorList input,                                                \
      at::TensorList tensors1,                                             \
      at::TensorList tensors2,                                             \
      const at::Tensor& scalars_) {                                        \
    auto scalars =                                                         \
        at::native::convert_tensor_to_scalar_list(scalars_, input.size()); \
    at::native::check_foreach_api_restrictions(                            \
        input, tensors1, tensors2, scalars);                               \
    if (!at::native::can_use_fast_route({input, tensors1, tensors2}) ||    \
        at::native::has_integral_tensor(input, /* includeBool */ true)) {  \
      return at::native::foreach_tensor_##NAME##_scalarlist_slow_(         \
          input, tensors1, tensors2, scalars);                             \
    }                                                                      \
                                                                           \
    FUNC##_(input, tensors1, tensors2, scalars, MODE);                     \
  }

FOREACH_POINTWISE_OP_SCALAR(
    addcmul,
    CNNL_FOREACH_ADDCMUL,
    foreach_pointwise_scalar);
FOREACH_POINTWISE_OP_SCALAR(
    addcdiv,
    CNNL_FOREACH_ADDCDIV,
    foreach_pointwise_scalar);
FOREACH_POINTWISE_OP_SCALARLIST(
    addcmul,
    CNNL_FOREACH_ADDCMUL,
    foreach_pointwise_scalarlist);
FOREACH_POINTWISE_OP_SCALARLIST(
    addcdiv,
    CNNL_FOREACH_ADDCDIV,
    foreach_pointwise_scalarlist);
FOREACH_POINTWISE_OP_TENSOR(
    addcmul,
    CNNL_FOREACH_ADDCMUL,
    foreach_pointwise_scalarlist);
FOREACH_POINTWISE_OP_TENSOR(
    addcdiv,
    CNNL_FOREACH_ADDCDIV,
    foreach_pointwise_scalarlist);

#undef FOREACH_POINTWISE_OP_SCALAR
#undef FOREACH_POINTWISE_OP_SCALARLIST
#undef FOREACH_POINTWISE_OP_TENSOR
} // namespace torch_mlu::ops
