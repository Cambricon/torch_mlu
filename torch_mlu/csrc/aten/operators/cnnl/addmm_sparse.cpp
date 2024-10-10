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

#include <ATen/native/SparseTensorUtils.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

using namespace at::sparse;

// --------------------------------------------------------------------
// addmm(at::Tensor, SparseTensor, at::Tensor, at::Scalar, at::Scalar)
// --------------------------------------------------------------------
at::Tensor& s_addmm_out_sparse_dense_mlu(
    at::Tensor& r,
    const at::Tensor& t,
    const SparseTensor& sparse,
    const at::Tensor& dense,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  TORCH_CHECK(
      t.device().is_privateuseone(),
      "Expected all tensors to be on the same device. addmm: expected 'self' to be MLU, but got CPU");
  TORCH_CHECK(
      r.device().is_privateuseone(),
      "Expected all tensors to be on the same device. addmm: expected 'out' to be MLU, but got CPU");
  TORCH_CHECK(
      sparse.device().is_privateuseone(),
      "Expected all tensors to be on the same device. addmm: expected 'mat1' to be MLU, but got CPU");
  TORCH_CHECK(
      dense.device().is_privateuseone(),
      "Expected all tensors to be on the same device. addmm: expected 'mat2' to be MLU, but got CPU");

  TORCH_CHECK(torch_mlu::check_device({sparse, r, t, dense}));

  TORCH_CHECK(
      dense.dim() == 2,
      "addmm: 2D tensor expected, got ",
      dense.dim(),
      "D tensor");
  TORCH_CHECK(
      sparse.sparse_dim() == 2,
      "addmm: expected first two dims to be sparse (indices has size 2 at first dim), but got ",
      sparse.sparse_dim(),
      " sparse dims");
  // no need to check dense_dim because dense_dim + sparse_dim = dim

  // mxk * kxn = mxn
  int64_t m = sparse.size(0);
  int64_t k = sparse.size(1);
  int64_t n = dense.size(1);

  TORCH_CHECK(
      t.size(0) == m,
      "addmm: Argument #1 (t): Expected dim 0 size ",
      m,
      ", got ",
      t.size(0));
  TORCH_CHECK(
      t.size(1) == n,
      "addmm: Argument #1 (t): Expected dim 1 size ",
      n,
      ", got ",
      t.size(1));
  TORCH_CHECK(
      dense.size(0) == k,
      "addmm: Argument #3 (dense): Expected dim 0 size ",
      k,
      ", got ",
      dense.size(0));

  r.resize_({m, n});

  SparseTensor coalesced_sparse = sparse;
  if (!sparse.is_coalesced()) {
    coalesced_sparse = sparse.coalesce();
  }

  int64_t nnz = coalesced_sparse._nnz();
  if (nnz == 0) {
    at::mul_out(r, t, at::scalar_tensor(beta, r.options()));
    return r;
  }

  at::Tensor indices = coalesced_sparse._indices();
  at::Tensor values = coalesced_sparse._values();
  at::Tensor indices_contiguous = cnnl_contiguous(indices);
  at::Tensor values_contiguous = cnnl_contiguous(values);
  at::Tensor row_indices = indices_contiguous.select(0, 0);
  at::Tensor col_indices = indices_contiguous.select(0, 1);

  bool is_trans_dense = false;
  at::Tensor dense_contiguous = cnnl_contiguous(dense);

  float cast_beta = beta.to<float>();
  if (cast_beta == 0) {
    r.zero_();
  } else {
    r.copy_(t);
  }

  auto r_contiguous = cnnl_contiguous(r);

  cnnl_addmm_sparse_out_internal(
      r_contiguous,
      row_indices,
      col_indices,
      values_contiguous,
      dense_contiguous,
      nnz,
      m,
      n,
      k,
      /*is_trans_sparse*/ false,
      is_trans_dense,
      beta,
      alpha,
      /*allow_tf32*/ false);

  if (!r.is_same(r_contiguous)) {
    r.copy_(r_contiguous);
  }

  return r;
}

at::Tensor& cnnl_addmm_out_sparse(
    const at::Tensor& self,
    const SparseTensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  c10::MaybeOwned<at::Tensor> b_self =
      expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_dense_mlu(result, *b_self, mat1, mat2, beta, alpha);
}

at::Tensor s_addmm_sparse_dense_mlu(
    const at::Tensor& t,
    const SparseTensor& sparse,
    const at::Tensor& dense,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  at::Tensor r = at::empty({0}, t.options());
  s_addmm_out_sparse_dense_mlu(r, t, sparse, dense, beta, alpha);
  return r;
}

at::Tensor cnnl_addmm_sparse(
    const at::Tensor& self,
    const SparseTensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  c10::MaybeOwned<at::Tensor> b_self =
      expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_sparse_dense_mlu(*b_self, mat1, mat2, beta, alpha);
}

at::Tensor& cnnl_addmm__sparse(
    at::Tensor& t,
    const SparseTensor& sparse,
    const at::Tensor& dense,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return s_addmm_out_sparse_dense_mlu(t, t, sparse, dense, beta, alpha);
}

} // namespace ops
} // namespace torch_mlu
