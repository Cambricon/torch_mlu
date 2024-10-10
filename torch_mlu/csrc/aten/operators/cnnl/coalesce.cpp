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
#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/native/SparseTensorUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include <bitset>
#include <memory>

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {
using namespace at::sparse;

std::set<at::ScalarType> coalesce_support_dtype{
    at::ScalarType::Half,
    at::ScalarType::Float};

SparseTensor cnnl__coalesce_sparse(const SparseTensor& self) {
  TORCH_CHECK(
      coalesce_support_dtype.find(self.scalar_type()) !=
          coalesce_support_dtype.end(),
      "MLU coalesce op not implemented for '",
      self.scalar_type(),
      "'");

  if (self.is_coalesced()) {
    return self;
  }
  int64_t nnz = self._nnz();
  // coalesce is not an in-place operation when is_coalesced is false,
  // we should keep the original tensor intact and do coalesce on a copy of the
  // tensor
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  // indices:[sparse_dims, nnz]
  // values: [nnz, dense_dims]
  // inverse_indices[nnz, ]
  // embedding_output [new_nnz, dense_dim]
  at::Tensor values = self._values();
  at::Tensor indices = self._indices();
  auto memory_format = values.suggest_memory_format();
  auto values_contiguous = cnnl_contiguous(values, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  // step1: call cnnl unique to remove duplicates indice
  int64_t dim = indices.dim() - 1;
  at::Tensor new_indices;
  at::Tensor inverse_indices =
      at::empty(indices_contiguous.size(dim), indices.options());
  std::tie(new_indices, inverse_indices, std::ignore) = cnnl_unique_internal(
      indices_contiguous.to(at::kInt),
      dim,
      /*return_inverse*/ true,
      /*return_counts*/ false);

  // step2: call cnnl embedding_backward to accumulate values corresponding to
  // the same index.
  at::Tensor grad_data = values_contiguous;
  if (grad_data.dim() == 1) {
    grad_data = grad_data.unsqueeze(1);
  }

  int64_t new_nnz = new_indices.size(1);
  // when the input tensor is empty, cnnl unique will set output_num to 0.
  if (new_nnz == 0) {
    // when indices is empty, coalesce will set nnz to 1 and all values will be
    // accumulated.
    new_nnz = 1;
    inverse_indices =
        at::zeros(indices_contiguous.size(dim), indices.options());
    std::vector<int64_t> indices_shape = indices.sizes().vec();
    indices_shape.back() = new_nnz;
    new_indices.resize_(indices_shape);
  }

  at::Tensor new_values;
  if (values.dim() == 1) {
    new_values = at::empty({new_nnz, 1}, values.options());
  } else {
    std::vector<int64_t> shape = {new_nnz};
    shape.insert(shape.end(), values.sizes().begin() + 1, values.sizes().end());
    new_values = at::empty(at::IntArrayRef(shape), values.options());
  }

  cnnl_embedding_dense_backward_internal(
      /*grad_output*/ grad_data,
      /*indices*/ inverse_indices.to(at::kInt),
      /*num_weights*/ 0,
      /*padding_idx*/ -1,
      /*scale_grad_by_freq*/ false,
      /*output*/ new_values);
  if (values.dim() == 1) {
    new_values.resize_((new_nnz));
  }

  SparseTensor dst = at::_sparse_coo_tensor_unsafe(
                         new_indices.to(at::kLong), new_values, self.sizes())
                         ._coalesced_(true);
  return dst;
}

} // namespace ops
} // namespace torch_mlu
