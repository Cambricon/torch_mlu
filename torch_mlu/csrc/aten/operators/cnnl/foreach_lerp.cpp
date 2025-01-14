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

// ternary_ops
::std::vector<at::Tensor> cnnl__foreach_lerp(
    at::TensorList self,
    at::TensorList tensors1,
    at::TensorList weights) {
  at::native::check_foreach_api_restrictions(self, tensors1, weights);
  if (!at::native::can_use_fast_route({self, tensors1, weights}, {}, true)) {
    return at::native::foreach_tensor_ternary_lerp_slow(
        self, tensors1, weights);
  }
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(self.size());
  for (const auto& t : self) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "cnnl__foreach_lerp_tensor_list",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_lerp_op<opmath_t, false>(
            self,
            tensors1,
            weights,
            vec_res,
            {},
            1.0,
            cnnlForeachLerpMode_t::FOREACH_LERP_TENSOR_LIST);
      });
  return vec_res;
}

void cnnl__foreach_lerp_(
    at::TensorList self,
    at::TensorList tensors1,
    at::TensorList weights) {
  at::native::check_foreach_api_restrictions(self, tensors1, weights);
  if (!at::native::can_use_fast_route({self, tensors1, weights}, {}, true)) {
    return at::native::foreach_tensor_ternary_lerp_slow_(
        self, tensors1, weights);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "cnnl__foreach_lerp_tensor_list_",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_lerp_op<opmath_t, true>(
            self,
            tensors1,
            weights,
            self,
            {},
            1.0,
            cnnlForeachLerpMode_t::FOREACH_LERP_TENSOR_LIST);
      });
  increment_version(self);
}

// scalar
::std::vector<at::Tensor> cnnl__foreach_lerp(
    at::TensorList self,
    at::TensorList tensors1,
    const at::Scalar& weight) {
  at::native::check_foreach_api_restrictions(self, tensors1);
  if (!at::native::can_use_fast_route({self, tensors1}, {}, true)) {
    return at::native::foreach_tensor_lerp_list_kernel_slow(
        self, tensors1, weight);
  }
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(self.size());
  for (const auto& t : self) {
    vec_res.emplace_back(at::native::empty_like(t));
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "cnnl__foreach_lerp_scalar",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_lerp_op<opmath_t, false>(
            self,
            tensors1,
            {},
            vec_res,
            {},
            weight.to<opmath_t>(),
            cnnlForeachLerpMode_t::FOREACH_LERP_SCALAR);
      });
  return vec_res;
}

void cnnl__foreach_lerp_(
    at::TensorList self,
    at::TensorList tensors1,
    const at::Scalar& weight) {
  at::native::check_foreach_api_restrictions(self, tensors1);
  if (!at::native::can_use_fast_route({self, tensors1}, {}, true)) {
    return at::native::foreach_tensor_lerp_list_kernel_slow_(
        self, tensors1, weight);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self[0].scalar_type(),
      "cnnl__foreach_lerp_scalar_",
      [&]() {
        using opmath_t = torch_mlu::MLUOpMathType_t<scalar_t>;
        cnnl_foreach_lerp_op<opmath_t, true>(
            self,
            tensors1,
            {},
            self,
            {},
            weight.to<opmath_t>(),
            cnnlForeachLerpMode_t::FOREACH_LERP_SCALAR);
      });
  increment_version(self);
}

} // namespace torch_mlu::ops
