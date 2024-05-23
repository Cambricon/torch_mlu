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

/**
 * Note [beta and alpha type in matmul ops]
 *
 * For mm, bmm, addmm, addbmm, addmv, addr, baddbmm ops, beta and alpha type
 * for cnnl kernel is a little different with gpu side. GPU using float when
 * input type is float, half, BFloat16; and using double when input
 * type is double. MLU side always using float. And more specific CNNL kernel
 * names are cnnlBatchMatMulBCast_v2 and cnnlMatMul_v2.
 */

void cnnl_addmm_out_internal(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    bool is_trans_self_,
    bool is_trans_mat1_,
    bool is_trans_mat2_,
    const at::Scalar& beta_,
    const at::Scalar& alpha_,
    bool allow_tf32_) {
  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto mat1_impl = getMluTensorImpl(mat1);
  auto mat2_impl = getMluTensorImpl(mat2);
  auto result_impl = getMluTensorImpl(result);

  // create desc
  CnnlMatmulDescriptor matmul_desc;
  CnnlMatmulAlgorithm matmul_algo;
  cnnlMatMulPrefer_t preference;
  CnnlMatmulHeuristicResult matmul_hr;
  int return_algo_count;
  int requested_algo_count = 1;
  int32_t is_trans_self = is_trans_self_;
  int32_t is_trans_mat1 = is_trans_mat1_;
  int32_t is_trans_mat2 = is_trans_mat2_;
  int32_t allow_tf32 = allow_tf32_;
  int64_t ldc = result.strides()[0];
  int64_t lda = mat1.strides()[0];
  int64_t ldb = mat2.strides()[0];
  int64_t m = is_trans_mat1_ ? mat1.sizes()[1] : mat1.sizes()[0];
  int64_t k = is_trans_mat1_ ? mat1.sizes()[0] : mat1.sizes()[1];
  int64_t n = is_trans_mat2_ ? mat2.sizes()[0] : mat2.sizes()[1];
  if (m <= 1) {
    ldc = std::max<int64_t>(n, 1);
  }

  if (is_trans_mat1_) {
    if (k <= 1) {
      lda = std::max<int64_t>(m, 1);
    }
  } else {
    if (m <= 1) {
      lda = std::max<int64_t>(k, 1);
    }
  }

  if (is_trans_mat2_) {
    if (n <= 1) {
      ldb = std::max<int64_t>(k, 1);
    }
  } else {
    if (k <= 1) {
      ldb = std::max<int64_t>(n, 1);
    }
  }

  matmul_desc.set_attr(CNNL_MATMUL_ALLOW_TF32, &(allow_tf32), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_DESC_TRANSA, &(is_trans_mat1), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_DESC_TRANSB, &(is_trans_mat2), sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_DESC_LDA, &(lda), sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_DESC_LDB, &(ldb), sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_DESC_LDC, &(ldc), sizeof(int32_t));

  if (beta_.to<float>() != 0.0f) {
    // for addmm
    int32_t use_beta = 1;
    matmul_desc.set_attr(CNNL_MATMUL_USE_BETA, &(use_beta), sizeof(int32_t));
  }

  // TODO(xushuo): CNNL_MATMUL_DESC_TRANSC will be supported in the future
  // get descriptor
  auto input_cnnl_type = getCnnlType(self_impl);
  auto self_desc = getTensorDesc(self_impl, input_cnnl_type);
  auto mat1_desc = getTensorDesc(mat1_impl, input_cnnl_type);
  auto mat2_desc = getTensorDesc(mat2_impl, input_cnnl_type);
  auto result_desc = getTensorDesc(result_impl, input_cnnl_type);

  auto handle = getCurrentHandle();

  matmul_hr.get(
      handle,
      matmul_desc.desc(),
      mat1_desc.get(),
      mat2_desc.get(),
      self_desc.get(),
      result_desc.get(),
      preference,
      requested_algo_count,
      &return_algo_count);

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetMatMulHeuristicResult(
      matmul_hr.hr(), matmul_algo.mut_algo(), &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  auto mat1_ptr = mat1_impl->mlu_data_ptr();
  auto mat2_ptr = mat2_impl->mlu_data_ptr();
  auto self_ptr = self_impl->mlu_data_ptr();
  auto result_ptr = result_impl->mlu_data_ptr();

  // bf16, complex are not supported
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "MLU mm",
      [&] {
        // More details in Note [beta and alpha type in matmul ops]
        using math_type = MLUAccumulateType_t<scalar_t>;
        auto alpha = alpha_.to<math_type>();
        auto beta = beta_.to<math_type>();
        TORCH_CNNL_CHECK(cnnlMatMul_v2(
            handle,
            matmul_desc.desc(),
            matmul_algo.algo(),
            &alpha,
            mat1_desc.get(),
            mat1_ptr,
            mat2_desc.get(),
            mat2_ptr,
            &beta,
            self_desc.get(),
            self_ptr,
            workspace_ptr.get(),
            workspace_size,
            result_desc.get(),
            result_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
