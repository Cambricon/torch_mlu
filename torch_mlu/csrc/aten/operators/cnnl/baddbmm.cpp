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

std::tuple<at::Tensor, bool> getBMMInput(const at::Tensor& self) {
  TORCH_CHECK(self.dim() == 3, "dimension must be 3 in bmm.");
  bool is_trans_self;
  if ((!self.is_contiguous()) && self.is_non_overlapping_and_dense()) {
    auto permute_back_order = get_permute_back_order(self);
    at::IntArrayRef back_array_order(permute_back_order);
    auto self_before_permute = self.permute(back_array_order);
    TORCH_CHECK(
        self_before_permute.is_contiguous(),
        "error order in permute_back_order.");
    int64_t batch_order = 0;
    for (int64_t i = 0; i < self.dim(); ++i) {
      if (permute_back_order[i] == 0) {
        batch_order = i;
        permute_back_order[i] = permute_back_order[0];
        permute_back_order[0] = 0;
      }
    }
    auto self_contiguous = torch_mlu::ops::cnnl_transpose_internal(
        self_before_permute, batch_order, 0);
    TORCH_CHECK(
        self_contiguous.is_contiguous(),
        "output must be contiguous in getBMMInput.");
    if (permute_back_order[1] == 1) {
      is_trans_self = false;
    } else {
      is_trans_self = true;
    }
    return std::make_tuple(self_contiguous, is_trans_self);
  } else {
    is_trans_self = false;
    return std::make_tuple(
        cnnl_contiguous(self, c10::MemoryFormat::Contiguous), is_trans_self);
  }
}

const at::Tensor& baddbmm_out_mlu_impl(
    const at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  // NB: when baddbmm and beta != 0.0, result has already
  // filled with self in meta function. So we do not need
  // handle it, and self is not used.

  at::IntArrayRef batch1_sizes = batch1.sizes();
  if (result.numel() == 0) {
    return result;
  } else if (batch1_sizes[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      return result.mul_(beta);
    }
  }

  at::ScalarType scalar_type = self.scalar_type();
  TORCH_CHECK(
      scalar_type == batch1.scalar_type(),
      "expected scalar type ",
      scalar_type,
      " but found ",
      batch1.scalar_type());
  TORCH_CHECK(
      scalar_type == batch2.scalar_type(),
      "expected scalar type ",
      scalar_type,
      " but found ",
      batch2.scalar_type());
  TORCH_CHECK(
      scalar_type == result.scalar_type(),
      "expected scalar type ",
      scalar_type,
      " but found ",
      result.scalar_type());

  // transposed output is not supported.
  auto result_contiguous = cnnl_contiguous(result);

  bool allow_tf32 = !at::NoTF32Guard::should_disable_tf32() &&
      torch_mlu::Global::instance().allowTF32CnMatMul();
  // cnnl does not support beta != 0 when use stride.
  if (beta.to<c10::complex<double>>() != 0.0) {
    // transpose of bmm
    bool is_trans_batch1;
    bool is_trans_batch2;
    at::Tensor batch1_contiguous;
    at::Tensor batch2_contiguous;

    std::tie(batch1_contiguous, is_trans_batch1) = getBMMInput(batch1);
    std::tie(batch2_contiguous, is_trans_batch2) = getBMMInput(batch2);
    cnnl_baddbmm_out_internal(
        result_contiguous,
        batch1_contiguous,
        batch2_contiguous,
        alpha,
        beta,
        is_trans_batch1,
        is_trans_batch2,
        allow_tf32);
  } else {
    cnnl_baddbmm_out_internal(
        result_contiguous,
        batch1,
        batch2,
        alpha,
        beta,
        // no need for transpose when use stride,
        // cnnl kernel will neglect transpose.
        false,
        false,
        allow_tf32);
  }

  if (!result.is_same(result_contiguous)) {
    result.copy_(result_contiguous);
  }
  return result;
}

TORCH_IMPL_FUNC(baddbmm_out_mlu)
(const at::Tensor& self,
 const at::Tensor& batch1,
 const at::Tensor& batch2,
 const at::Scalar& beta,
 const at::Scalar& alpha,
 const at::Tensor& result) {
  {
    at::NoNamesGuard guard;
    baddbmm_out_mlu_impl(result, self, batch1, batch2, beta, alpha);
  }
}

TORCH_IMPL_FUNC(bmm_out_mlu)
(const at::Tensor& batch1, const at::Tensor& batch2, const at::Tensor& result) {
  at::Scalar beta(0.0);
  at::Scalar alpha(1.0);
  {
    at::NoNamesGuard guard;
    baddbmm_out_mlu_impl(result, result, batch1, batch2, beta, alpha);
  }
}

} // namespace ops
} // namespace torch_mlu
