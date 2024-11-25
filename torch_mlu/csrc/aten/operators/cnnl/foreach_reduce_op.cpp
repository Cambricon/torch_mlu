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

#include <ATen/core/jit_type.h>
#include <ATen/native/ForeachUtils.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

namespace {
inline void check_foreach_norm_dtype(
    std::optional<at::ScalarType> opt_dtype,
    at::ScalarType self_dtype,
    const char* const name) {
  if (opt_dtype.has_value()) {
    auto dtype = opt_dtype.value();
    TORCH_CHECK(
        isFloatingType(dtype) || isComplexType(dtype),
        name,
        ": dtype should"
        " be floating point or complex, but got ",
        dtype);
    TORCH_CHECK(
        isComplexType(self_dtype) == isComplexType(dtype),
        name,
        ": dtype should be ",
        isComplexType(self_dtype) ? "complex" : "real",
        " for ",
        isComplexType(self_dtype) ? "complex" : "real",
        " inputs, but got ",
        dtype);
    TORCH_CHECK(
        promoteTypes(self_dtype, dtype) == dtype,
        name,
        ": the dtype of the input ",
        "(",
        self_dtype,
        ") should be convertible ",
        "without narrowing to the specified dtype (",
        dtype,
        ")");
  }
}
} // anonymous namespace

std::vector<at::Tensor> cnnl__foreach_norm(
    at::TensorList tensors,
    const at::Scalar& ord,
    std::optional<at::ScalarType> dtype) {
  const auto p = [&]() -> float {
    if (ord.isIntegral(false)) {
      return ord.to<int64_t>();
    } else if (ord.isFloatingPoint()) {
      return ord.to<float>();
    } else {
      TORCH_CHECK(
          false,
          "cnnl__foreach_tensor_norm expects ord to be integer or float");
    }
  }();
  at::native::check_foreach_api_restrictions(tensors);
  const bool has_int_or_complex =
      std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
            at::isComplexType(scalar_type);
      });
  if (!at::native::can_use_fast_route(tensors) || has_int_or_complex ||
      !(p == static_cast<float>(1) || p == static_cast<float>(2) ||
        p == std::numeric_limits<float>::infinity())) {
    return at::native::foreach_tensor_norm_slow(tensors, ord, dtype);
  }
  check_foreach_norm_dtype(
      dtype, tensors[0].scalar_type(), "cnnl__foreach_tensor_norm");

  const size_t ntensors = tensors.size();
  const auto options = tensors[0].options();
  const at::ScalarType output_dtype =
      dtype.has_value() ? dtype.value() : tensors[0].scalar_type();

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(ntensors);
  const auto res_option = options.dtype(output_dtype);
  for (const auto i : c10::irange(ntensors)) {
    vec_res.push_back(cnnl_empty(
        {},
        at::optTypeMetaToScalarType(res_option.dtype_opt()),
        res_option.layout_opt(),
        res_option.device_opt(),
        res_option.pinned_memory_opt(),
        res_option.memory_format_opt()));
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "cnnl_foreach_norm_internal",
      [&]() { cnnl_foreach_norm_internal(tensors, vec_res, p); });

  std::vector<at::Tensor> result;
  result.reserve(ntensors);
  for (int i = 0; i < ntensors; i++) {
    if (tensors[i].numel() != 0) {
      result.emplace_back(vec_res[i]);
    } else {
      result.emplace_back(at::zeros({}, res_option));
    }
  }
  return result;
}

} // namespace ops
} // namespace torch_mlu
