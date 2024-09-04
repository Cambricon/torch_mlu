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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// CNNL limits:
// When is_dim_none is false, the total size of input in dimension dim should be
// less than 167936 bytes. When is_dim_none is true, the total size of input
// should be less than 671744 bytes.
static void median_shape_check(
    const at::Tensor& input,
    int64_t dim,
    bool is_dim_none) {
  int64_t half_max_size = (is_dim_none) ? 335872 : 83968;
  int64_t float_max_size = (is_dim_none) ? 167936 : 41984;
  std::unordered_map<at::ScalarType, int64_t> m = {
      {at::ScalarType::Half, half_max_size},
      {at::ScalarType::BFloat16, half_max_size},
      {at::ScalarType::Float, float_max_size},
      {at::ScalarType::Double, float_max_size},
  };
  TORCH_CHECK(
      m.count(input.scalar_type()) > 0,
      "MLU median not implemented for ",
      input.scalar_type());
  int64_t size_dim = (is_dim_none) ? input.numel() : input.sizes()[dim];
  if (is_dim_none) {
    TORCH_CHECK(
        size_dim < m[input.scalar_type()],
        "MLU median: elements number of input tensor ",
        "should be less than ",
        m[input.scalar_type()]);
  } else {
    TORCH_CHECK(
        size_dim < m[input.scalar_type()],
        "MLU median: elements number of input tensor in the dimension ",
        dim,
        " should be less than ",
        m[input.scalar_type()]);
  }
}

at::Tensor median_impl(const Tensor& self, bool ignore_nan) {
  at::NoNamesGuard guard;

  int64_t size = self.numel();
  // Return nan for empty tensors
  if (size <= 0) {
    return at::full({}, std::numeric_limits<float>::quiet_NaN())
        .to(self.options());
  }

  median_shape_check(self, 0, true);

  auto result = at::empty({}, self.options());
  auto indices = at::empty({0}, self.options().dtype(at::kInt));
  auto self_contiguous = cnnl_contiguous(self);
  cnnl_median_internal(
      self_contiguous, /*dim=*/0, result, indices, /*is_dim_none=*/true);
  return result;
}

std::tuple<at::Tensor&, at::Tensor&> median_with_indices_impl(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    bool ignore_nan) {
  at::NoNamesGuard guard;

  dim = at::maybe_wrap_dim(dim, self.dim());
  at::checkDeviceType("median", {values, indices}, self.device().type());
  at::checkScalarType("median", {indices, "indices", 1}, at::kLong);
  at::checkSameType("median", {values, "values", 0}, {self, "self", 2});

  std::vector<int64_t> out_shape = self.sizes().vec();
  at::native::zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    assert(dim >= 0);
    assert(dim < static_cast<int64_t>(out_shape.size()));

    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  values.resize_(out_shape);
  indices.resize_(out_shape);

  // Only launch kernel for non-empty tensors
  if (self.numel() > 0) {
    median_shape_check(self, dim, false);
    auto self_contiguous = cnnl_contiguous(self);
    auto values_contiguous = cnnl_contiguous(values);
    auto indices_contiguous =
        cast_long_to_int_if_needed(cnnl_contiguous(indices));
    cnnl_median_internal(
        self_contiguous,
        dim,
        values_contiguous,
        indices_contiguous,
        /*is_dim_none=*/false);
    if (!values.is_same(values_contiguous)) {
      values.copy_(values_contiguous);
    }
    if (!indices.is_same(indices_contiguous)) {
      indices.copy_(indices_contiguous);
    }
  }

  guard.reset();
  at::namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  at::namedinference::propagate_names_for_reduction(
      indices, self, dim, keepdim);

  return std::forward_as_tuple(values, indices);
}

at::Tensor cnnl_median(const at::Tensor& self) {
  return median_impl(self, /*ignore_nan=*/false);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_median_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/false);
}

} // namespace ops
} // namespace torch_mlu
