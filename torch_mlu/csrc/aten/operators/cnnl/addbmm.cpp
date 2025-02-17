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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

static void addbmm_impl_(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  TORCH_CHECK(
      batch1.size(0) == batch2.size(0),
      "batch1 and batch2 must have same number of batches, got ",
      batch1.size(0),
      " and ",
      batch2.size(0));
  TORCH_CHECK(
      batch1.size(2) == batch2.size(1),
      "Incompatible matrix sizes for bmm (",
      batch1.size(1),
      "x",
      batch1.size(2),
      " and ",
      batch2.size(1),
      "x",
      batch2.size(2),
      ")");

  const int64_t dim1 = batch1.size(1);
  const int64_t dim2 = batch2.size(2);
  TORCH_CHECK(
      self.size(0) == dim1 && self.size(1) == dim2,
      "self tensor does not match matmul output shape");

  result.resize_as_(self);

  if (beta.to<c10::complex<double>>() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  const int64_t num_batches = batch1.size(0);

  if (num_batches == 0) {
    if (beta.to<c10::complex<double>>() != 0.0) {
      result.mul_(beta);
    } else {
      result.zero_();
    }
    return;
  }

  // NB: We do not use same implementation with CUDA, because it is too slow.
  if (alpha.to<c10::complex<double>>() == 0.0) {
    result.mul_(beta);
  } else {
    auto bmm_result = at::bmm(batch1, batch2);
    bmm_result = bmm_result.sum({0}, false);
    auto self_contiguous = cnnl_contiguous(self);
    auto bmm_contiguous = cnnl_contiguous(bmm_result);
    auto result_contiguous = cnnl_contiguous(result);
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (beta.to<c10::complex<double>>() == 0.0) {
      result_contiguous = bmm_result;
    } else {
      cnnl_optensor_out_internal(
          result_contiguous,
          self_contiguous,
          bmm_contiguous,
          beta,
          alpha,
          0.0,
          CNNL_OP_TENSOR_ADD);
    }
    if (!result.is_same(result_contiguous)) {
      result.copy_(result_contiguous);
    }
  }
}

at::Tensor& cnnl_addbmm_out(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  auto b_self =
      at::expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");
  {
    at::NoNamesGuard guard;
    addbmm_impl_(result, *b_self, batch1, batch2, beta, alpha);
  }
  auto names =
      at::namedinference::propagate_names_for_addmm(batch1, batch2, self);
  at::namedinference::propagate_names_if_nonempty(result, names);
  return result;
}

at::Tensor& cnnl_addbmm_(
    at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return cnnl_addbmm_out(self, batch1, batch2, beta, alpha, self);
}

at::Tensor cnnl_addbmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  at::Tensor result = at::empty({0}, self.options());
  return cnnl_addbmm_out(self, batch1, batch2, beta, alpha, result);
}

} // namespace ops
} // namespace torch_mlu
