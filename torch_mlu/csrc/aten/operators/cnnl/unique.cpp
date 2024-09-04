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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_mlu_template(
    const at::Tensor& self,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  auto num_inp = self.numel();
  TORCH_CHECK(
      num_inp <= INT_MAX, "Large tensors are not supported by cnnl_unique");
  if (num_inp == 0) {
    at::Tensor output = at::empty({0}, self.options());
    at::Tensor inverse_indices =
        at::empty(self.sizes(), self.options().dtype(at::kLong));
    at::Tensor counts = at::empty({0}, self.options().dtype(at::kLong));
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        output, inverse_indices, counts);
  }
  auto self_contiguous = cnnl_contiguous(self);
  at::Tensor output, inverse, counts;
  // The current MLU implementation of unique always sort
  // to maintain consistency with CUDA. See the discussion in
  // https://github.com/pytorch/pytorch/issues/105742
  std::tie(output, inverse, counts) =
      cnnl_unique_internal(self_contiguous, -1, return_inverse, return_counts);
  return std::make_tuple(output, inverse, counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_mlu_template(
    const at::Tensor& self,
    const int64_t dim,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  auto num_inp = self.numel();
  TORCH_CHECK(
      num_inp <= INT_MAX, "Large tensors are not supported by cnnl_unique");
  auto sizes = self.sizes().vec();
  // check how many zero dimensions exist
  auto num_zero_dims = std::count(sizes.begin(), sizes.end(), 0);

  // tensor is not well formed as it has 0 sized dimensions
  if (self.size(dim) == 0) {
    TORCH_CHECK(
        num_zero_dims == 1,
        "Number of zero sized dimensions is more than one, so unique cannot be applied ")
    at::Tensor output = at::empty(sizes, self.options());
    at::Tensor inverse_indices =
        at::empty({0}, self.options().dtype(at::kLong));
    at::Tensor counts = at::empty({0}, self.options().dtype(at::kLong));

    return std::make_tuple(output, inverse_indices, counts);
  }

  TORCH_CHECK(
      num_zero_dims == 0,
      "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

  auto self_contiguous = cnnl_contiguous(self);
  // The current MLU implementation of unique always sort
  // to maintain consistency with CUDA. See the discussion in
  // https://github.com/pytorch/pytorch/issues/105742
  return cnnl_unique_internal(
      self_contiguous,
      at::maybe_wrap_dim(dim, self.dim()),
      return_inverse,
      return_counts);
}

std::tuple<at::Tensor, at::Tensor> cnnl__unique(
    const at::Tensor& self,
    const bool sorted,
    const bool return_inverse) {
  return AT_DISPATCH_MLU_FLOAT_HALF_AND_INT(self.scalar_type(), "unique", [&] {
    at::Tensor output, inverse;
    std::tie(output, inverse, std::ignore) =
        unique_mlu_template(self, sorted, return_inverse, false);
    return std::make_tuple(output, inverse);
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl__unique2(
    const at::Tensor& self,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  return AT_DISPATCH_MLU_FLOAT_HALF_AND_INT(self.scalar_type(), "unique", [&] {
    return unique_mlu_template(self, sorted, return_inverse, return_counts);
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_unique_dim(
    const at::Tensor& self,
    const int64_t dim,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  TORCH_CHECK(
      !(self.scalar_type() == c10::ScalarType::Half),
      "MLU unique does not support dim and half combinations.");
  return AT_DISPATCH_MLU_FLOAT_HALF_AND_INT(self.scalar_type(), "unique", [&] {
    return unique_dim_mlu_template(
        self, dim, sorted, return_inverse, return_counts);
  });
}

} // namespace ops
} // namespace torch_mlu
