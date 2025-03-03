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

#include <ATen/SparseTensorImpl.h>
#include <ATen/native/SparseTensorUtils.h>
#include "aten/utils/dispatch.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/mlu_ops_lite/sparse_add.h"

namespace torch_mlu {
namespace ops {

using namespace at::sparse;

// --------------------------------------------------------------------
// mul(SparseTensor, Scalar)
// --------------------------------------------------------------------

at::sparse::SparseTensor& mul_out_sparse_zerodim(
    at::sparse::SparseTensor& r,
    const at::sparse::SparseTensor& t,
    const Tensor& value) {
  AT_ASSERT(r.is_sparse());
  AT_ASSERT(t.is_sparse());
  AT_ASSERT(value.dim() == 0);

  // Resolve a possibly sparse COO value to a strided tensor.
  Tensor value_;
  if (value.is_sparse()) {
    if (value._nnz() == 0) {
      r.resize_as_(t);
      return r.zero_();
    }
    value_ = value.values();
  } else {
    value_ = value;
  }
  // With broadcasting in action, value_ may be a 1-D tensor as long
  // as its shape is (1,).
  AT_ASSERT(value_.numel() == 1);

  if (is_same_tensor(r, t)) {
    r._values().mul_(value_);
  } else {
    r.resize_as_(t);
    auto indices = r._indices();
    indices.resize_as_(t._indices());
    indices.copy_(t._indices());
    Tensor r_values = r._values();
    at::mul_out(r_values, t._values(), value_);
    get_sparse_impl(r)->set_nnz_and_narrow(t._nnz());
    r._coalesced_(t.is_coalesced());
  }
  return r;
}

at::sparse::SparseTensor& mul_out_sparse_scalar(
    at::sparse::SparseTensor& r,
    const at::sparse::SparseTensor& t,
    const at::Scalar& value) {
  return mul_out_sparse_zerodim(r, t, wrapped_scalar_tensor(value));
}

// dense + sparse
Tensor& add_out_dense_sparse_mlu(
    Tensor& r_,
    const Tensor& dense,
    const SparseTensor& sparse,
    const at::Scalar& value) {
  TORCH_CHECK(
      dense.is_privateuseone(),
      "add: expected 'self' to be a MLU tensor, but got a CPU tensor");
  TORCH_CHECK(
      sparse.is_privateuseone(),
      "add: expected 'other' to be a MLU tensor, but got a CPU tensor");
  TORCH_CHECK(
      r_.is_privateuseone(),
      "add: expected 'out' to be a MLU tensor, but got a CPU tensor");

  TORCH_CHECK(torch_mlu::check_device({sparse, r_, dense}));

  TORCH_CHECK(
      dense.sizes().equals(sparse.sizes()),
      "add: expected 'self' and 'other' to have same size, but self has size ",
      dense.sizes(),
      " while other has size ",
      sparse.sizes(),
      " (FYI: dense-sparse addition does not currently support broadcasting)");

  TORCH_CHECK(
      sparse.sparse_dim() <= 2,
      "mlu add dense + sparse now only support sparse_dim<=2");

  const int64_t nnz = sparse._nnz();
  if (nnz == 0) {
    r_.resize_as_(dense);
    r_.copy_(dense);
    return r_;
  }

  auto commonDtype = at::result_type(dense, sparse);
  TORCH_CHECK(
      at::canCast(commonDtype, r_.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      r_.scalar_type());

  Tensor r = r_;
  if (r_.scalar_type() != commonDtype) {
    r = at::empty_like(dense, r_.options().dtype(commonDtype));
  }

  Tensor dense_buffer = dense.to(commonDtype);
  Tensor values = sparse._values().to(commonDtype);

  // a little different with cuda, the copy operator was added in sparse_add.mlu
  if (!is_same_tensor(r, dense_buffer)) {
    r.resize_as_(dense);
  }

  if (values.numel() == 0) {
    return r_;
  }

  if (sparse.sparse_dim() == 0) {
    TORCH_CHECK(false, "sparse_dim=0 not supported");
  }

  Tensor indices = sparse._indices();
  Tensor indices_contiguous = cnnl_contiguous(indices);
  Tensor values_contiguous = cnnl_contiguous(values);
  Tensor dense_contiguous = cnnl_contiguous(dense_buffer);
  Tensor r_contiguous = cnnl_contiguous(r);
  void* indices_ptr = mlu_data_ptr(getMluTensorImpl(indices_contiguous));
  void* values_ptr = mlu_data_ptr(getMluTensorImpl(values_contiguous));
  void* dense_ptr = mlu_data_ptr(getMluTensorImpl(dense_contiguous));
  void* r_ptr = mlu_data_ptr(getMluTensorImpl(r_contiguous));

  // cnnl handle not mluop handle, it is used for cnnlCopy_v2
  // if r_contiguous is not dense_contiguous, copy dense_contiguous
  // to r_contiguous
  auto handle = getCurrentHandle();

  AT_DISPATCH_MLU_FLOAT_HALF_AND_BFLOAT16(
      commonDtype, "add_out_dense_sparse_mlu", [&] {
        torch_mlu::ops::mluSparseAddDense(
            handle,
            BANG_WRAP_T((int64_t*)indices_ptr),
            BANG_WRAP_T((scalar_t*)values_ptr),
            BANG_WRAP_T((scalar_t*)dense_ptr),
            value.to<float>(), // convert to scalar_t in .mlu
            nnz,
            r_contiguous.sizes().data(),
            sparse.sparse_dim(),
            sparse.dense_dim(),
            r_contiguous.numel(),
            BANG_WRAP_T((scalar_t*)r_ptr));
      });

  if (!r_.is_same(r_contiguous)) {
    r_.copy_(r_contiguous);
  }
  return r_;
}

SparseTensor& bang_add_out_sparse(
    const SparseTensor& t,
    const SparseTensor& src,
    const at::Scalar& value,
    SparseTensor& r_) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_mlu(r_, t, src, value);
  }

  TORCH_CHECK(
      src.is_sparse(),
      "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");

  TORCH_CHECK(
      t.is_privateuseone(), "add: expected 'self' to be MLU, but got CPU");
  TORCH_CHECK(
      src.is_privateuseone(), "add: expected 'other' to be MLU, but got CPU");
  TORCH_CHECK(
      r_.is_privateuseone(), "add: expected 'out' to be MLU, but got CPU");

  TORCH_CHECK(torch_mlu::check_device({r_, t, src}));

  auto commonDtype = at::result_type(t, src);
  TORCH_CHECK(
      canCast(commonDtype, r_.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      r_.scalar_type());

  TORCH_CHECK(
      t.sizes().equals(src.sizes()),
      "add: expected 'self' and 'other' to have same size, but ",
      t.sizes(),
      " != ",
      src.sizes());

  if (src._nnz() == 0) {
    return at::copy_sparse_to_sparse_(r_, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r_, src, value);
  }

  TORCH_CHECK(
      is_same_density(t, src),
      "add: expected 'self' and 'other' to have same density, but 'self' has ",
      t.sparse_dim(),
      " sparse dimensions while 'other' has ",
      src.sparse_dim(),
      " sparse dimensions");

  // We deliberately choose to simply concat the indices and values tensors
  // rather than merging them. This removes the need to synchronously fetch nnz
  // at the end of the operation, at the cost of having a non-coalesced result.
  // This trade-off is preferable for the common use-case of gradient
  // accumulation.
  Tensor t_indices_ = t._indices();
  Tensor s_indices_ = src._indices();

  Tensor t_values_ = t._values().to(commonDtype);
  Tensor s_values_ = src._values().to(commonDtype);

  AT_DISPATCH_MLU_FLOAT_HALF_INT_COMPLEX_AND_BFLOAT16(
      commonDtype, "add_out_sparse_mlu", [&] {
        if (value.to<scalar_t>() != scalar_t(1)) {
          s_values_ = s_values_.mul(value);
        }
      });
  Tensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
  Tensor r_values_ = at::cat({t_values_, s_values_}, 0);

  if (r_.scalar_type() != commonDtype) {
    SparseTensor promoted = at::empty({0}, r_.options().dtype(commonDtype));
    promoted.resize_as_(src);
    alias_into_sparse(promoted, r_indices_, r_values_);
    // performs the addition under the common dtype.
    promoted = promoted.coalesce();
    r_values_ = promoted._values().to(r_.scalar_type());
    r_indices_ = promoted._indices();
  } else {
    r_.resize_as_(src);
  }

  alias_into_sparse(r_, r_indices_, r_values_);

  // Prevent unbounded growth of nnz
  // TODO: Improved heuristic on when to coalesce or remove need to coalesce
  if (r_._nnz() > r_.numel()) {
    auto c = r_.coalesce();
    alias_into_sparse(r_, c._indices(), c._values());
  }

  return r_;
}

} // namespace ops
} // namespace torch_mlu
