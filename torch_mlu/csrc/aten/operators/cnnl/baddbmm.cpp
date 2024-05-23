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

c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(
    const Tensor& tensor,
    bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> prepare_batch_matrix_for_cnnl(
    const Tensor& tensor,
    bool& transpose_tensor,
    int32_t& ld_tensor,
    bool transpose_result,
    int32_t m,
    int32_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  c10::MaybeOwned<Tensor> tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;
  if (tensor_strides[fast_dim] == 1 &&
      (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = true;
    tensor_ = resolve_conj_if_indicated(tensor, false);
    ld_tensor = tensor_->strides()[leading_dim];
  } else if (
      (tensor_strides[leading_dim] == 1) &&
      (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = false;
    tensor_ = resolve_conj_if_indicated(tensor, true);
    ld_tensor = tensor_->strides()[fast_dim];
  } else {
    transpose_tensor = transpose_result;
    // gemm call requires leading dimension and stride parameters to be non-zero
    bool is_stride_non_zero =
        tensor.strides()[1] != 0 && tensor.strides()[2] != 0;
    if (tensor.is_contiguous() && is_stride_non_zero) {
      tensor_ = resolve_conj_if_indicated(tensor, transpose_result);
    } else {
      tensor_ = c10::MaybeOwned<Tensor>::owned(
          tensor.clone(at::MemoryFormat::Contiguous));
    }
    ld_tensor = tensor_->strides()[1];
  }

  return tensor_;
}

const at::Tensor& baddbmm_out_mlu_impl(
    const at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1.size(2) == 0) {
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

  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) ||
       (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    transpose_result = true;
    result_ = resolve_conj_if_indicated(result, true);
  } else if (
      (result_strides[2] == 1) &&
      (result_sizes[1] == 1 ||
       (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = false;
    result_ = resolve_conj_if_indicated(result, true);
  } else {
    result_ = c10::MaybeOwned<Tensor>::owned(
        result.clone(at::MemoryFormat::Contiguous));
  }

  int leading_dim = transpose_result ? 1 : 2;
  int fast_dim = transpose_result ? 2 : 1;

  int32_t m = result_sizes[transpose_result ? 2 : 1];
  int32_t n = result_sizes[leading_dim];
  int32_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int32_t lda, ldb, ldc;
  bool transpose_batch1, transpose_batch2;
  auto batch1_ = prepare_batch_matrix_for_cnnl(
      transpose_result ? batch2 : batch1,
      transpose_batch1,
      lda,
      transpose_result,
      m,
      k);
  auto batch2_ = prepare_batch_matrix_for_cnnl(
      transpose_result ? batch1 : batch2,
      transpose_batch2,
      ldb,
      transpose_result,
      k,
      n);
  ldc = result_->strides()[fast_dim];
  int32_t num_batches = result_->sizes()[0];
  bool allow_tf32 = !at::NoTF32Guard::should_disable_tf32() &&
      at::globalContext().allowTF32CnMatMul();

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (m <= 1) {
    ldc = std::max<int64_t>(n, 1);
  }

  if (transpose_batch1) {
    if (k <= 1) {
      lda = std::max<int64_t>(m, 1);
    }
  } else {
    if (m <= 1) {
      lda = std::max<int64_t>(k, 1);
    }
  }

  if (transpose_batch2) {
    if (n <= 1) {
      ldb = std::max<int64_t>(k, 1);
    }
  } else {
    if (k <= 1) {
      ldb = std::max<int64_t>(n, 1);
    }
  }

  // complex are not supported
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      result.scalar_type(),
      "MLU bmm",
      [&] {
        at::Tensor batch1_tensor = *batch1_;
        at::Tensor batch2_tensor = *batch2_;
        at::Tensor result_tensor = *result_;
        // If batch is 1 call gemm rather than bgemm
        if (num_batches == 1) {
          at::Tensor d_tensor = transpose_result
              ? at::squeeze(result_tensor, 0).t()
              : at::squeeze(result_tensor, 0);
          cnnl_addmm_out_internal(
              d_tensor,
              d_tensor,
              (transpose_result == transpose_batch1)
                  ? at::squeeze(batch1_tensor, 0)
                  : at::squeeze(batch1_tensor, 0).t(),
              (transpose_result == transpose_batch2)
                  ? at::squeeze(batch2_tensor, 0)
                  : at::squeeze(batch2_tensor, 0).t(),
              transpose_result,
              transpose_batch1,
              transpose_batch2,
              beta,
              alpha,
              allow_tf32);
        } else {
          cnnl_baddbmm_out_internal(
              transpose_batch1,
              transpose_batch2,
              m,
              n,
              k,
              num_batches,
              result_tensor,
              ldc,
              result_->strides()[0],
              alpha,
              batch1_tensor,
              lda,
              batch1_->strides()[0],
              batch2_tensor,
              ldb,
              batch2_->strides()[0],
              beta,
              const_cast<at::Tensor&>(result_tensor),
              ldc,
              result_->strides()[0],
              allow_tf32);
        }
      });

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
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
