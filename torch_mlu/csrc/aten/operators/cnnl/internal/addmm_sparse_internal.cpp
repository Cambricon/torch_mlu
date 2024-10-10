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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_addmm_sparse_out_internal(
    at::Tensor& result,
    const at::Tensor& mat1_row_indices,
    const at::Tensor& mat1_col_indices,
    const at::Tensor& mat1_values,
    const at::Tensor& mat2,
    int nnz,
    int m,
    int n,
    int k,
    bool is_trans_mat1_,
    bool is_trans_mat2_,
    const at::Scalar& beta_,
    const at::Scalar& alpha_,
    bool allow_tf32_) {
  auto row_indices_impl = getMluTensorImpl(mat1_row_indices);
  auto row_indices_desc = getTensorDesc(row_indices_impl);
  auto row_indices_ptr = mlu_data_ptr(row_indices_impl);

  auto col_indices_impl = getMluTensorImpl(mat1_col_indices);
  auto col_indices_desc = getTensorDesc(col_indices_impl);
  auto col_indices_ptr = mlu_data_ptr(col_indices_impl);

  auto values_impl = getMluTensorImpl(mat1_values);
  auto values_desc = getTensorDesc(values_impl);
  auto values_ptr = mlu_data_ptr(values_impl);

  auto mat2_impl = getMluTensorImpl(mat2);
  auto mat2_desc = getTensorDesc(mat2_impl);
  auto mat2_ptr = mlu_data_ptr(mat2_impl);

  auto result_impl = getMluTensorImpl(result);
  auto result_desc = getTensorDesc(result_impl);
  auto result_ptr = mlu_data_ptr(result_impl);

  // create sparse tensor desc
  auto mat1_desc = getSparseCOOTensorDesc(
      m,
      k,
      nnz,
      row_indices_desc.get(),
      row_indices_ptr,
      col_indices_desc.get(),
      col_indices_ptr,
      values_desc.get(),
      values_ptr);

  CnnlSparseDenseMatmulDescriptor sp_matmul_desc;
  cnnlDataType_t compute_type = getCnnlDataType(mat2.dtype());
  sp_matmul_desc.set_attr(
      CNNL_SPARSE_DENSE_MATMUL_DESC_COMPUTE_TYPE,
      &(compute_type),
      sizeof(compute_type));
  sp_matmul_desc.set_attr(
      CNNL_SPARSE_DENSE_MATMUL_DESC_TRANSA,
      &(is_trans_mat1_),
      sizeof(is_trans_mat1_));
  sp_matmul_desc.set_attr(
      CNNL_SPARSE_DENSE_MATMUL_DESC_TRANSB,
      &(is_trans_mat2_),
      sizeof(is_trans_mat2_));

  AT_DISPATCH_FLOATING_TYPES(mat2.scalar_type(), "MLU addmm_sparse", [&] {
    auto alpha = alpha_.to<scalar_t>();
    auto beta = beta_.to<scalar_t>();
    auto handle = getCurrentHandle();
    cnnlSparseDenseMatmulAlgo_t algo = CNNL_SPMM_ALGO_0;
    size_t workspace_size;
    TORCH_CNNL_CHECK(cnnlGetSparseDenseMatmulWorkspaceSize(
        handle,
        sp_matmul_desc.desc(),
        algo,
        &alpha,
        mat1_desc.get(),
        mat2_desc.get(),
        &beta,
        result_desc.get(),
        &workspace_size));
    auto workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
    TORCH_CNNL_CHECK(cnnlSparseDenseMatmul(
        handle,
        sp_matmul_desc.desc(),
        algo,
        &alpha,
        mat1_desc.get(),
        mat2_desc.get(),
        mat2_ptr,
        &beta,
        result_desc.get(),
        result_ptr,
        workspace_ptr.get(),
        workspace_size));
  });
}

} // namespace ops
} // namespace torch_mlu
