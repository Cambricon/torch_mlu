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

void cnnl_addmm_bias_out_internal(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    bool is_trans_self_,
    bool is_trans_mat1_,
    bool is_trans_mat2_,
    const at::Scalar& beta_,
    const at::Scalar& alpha_,
    cnnlActivationMode_t mode,
    bool allow_tf32_) {
  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto self_desc = getTensorDesc(self_impl);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto mat1_impl = getMluTensorImpl(mat1);
  auto mat1_desc = getTensorDesc(mat1_impl);
  auto mat1_ptr = mlu_data_ptr(mat1_impl);

  auto mat2_impl = getMluTensorImpl(mat2);
  auto mat2_desc = getTensorDesc(mat2_impl);
  auto mat2_ptr = mlu_data_ptr(mat2_impl);

  auto result_impl = getMluTensorImpl(result);
  auto result_desc = getTensorDesc(result_impl);
  auto result_ptr = mlu_data_ptr(result_impl);

  // create desc
  CnnlMatmulExDescriptor matmul_desc;
  CnnlMatmulExAlgorithm matmul_algo;
  cnnlMatMulExPrefer_t preference;
  CnnlMatmulExHeuristicResult matmul_hr;

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

  auto compute_cnnl_type = CNNL_DTYPE_FLOAT;

  matmul_desc.set_attr(
      CNNL_MATMUL_EX_DESC_COMPUTE_TYPE,
      &(compute_cnnl_type),
      sizeof(compute_cnnl_type));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_ALLOW_TF32, &(allow_tf32), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_DESC_TRANSA, &(is_trans_mat1), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_DESC_TRANSB, &(is_trans_mat2), sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_EX_DESC_LDA, &(lda), sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_EX_DESC_LDB, &(ldb), sizeof(int32_t));
  matmul_desc.set_attr(CNNL_MATMUL_EX_DESC_LDC, &(ldc), sizeof(int32_t));

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  cnnlNanPropagation_t nan_prop = CNNL_NOT_PROPAGATE_NAN;

  // set activation info.
  CnnlActivationDescriptor desc_activation;
  desc_activation.set(mode, prefer, nan_prop, 0, 0, 0, 0, false, true);

  // bias => self*beta(1.0) => self
  cnnlMatMulEpilogueType_t epilogue_type =
      CNNL_MATMUL_EPI_BIAS_SCALE_BN_ACTIVATION;
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_DESC_EPILOGUE_TYPE,
      &(epilogue_type),
      sizeof(epilogue_type));
  TORCH_CNNL_CHECK(cnnlSetMatMulExBiasScaleBNActive(
      matmul_desc.desc(),
      self_desc.get(),
      self_ptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      0,
      desc_activation.desc()));

  auto handle = getCurrentHandle();
  matmul_hr.get(
      handle,
      matmul_desc.desc(),
      mat1_desc.get(),
      mat2_desc.get(),
      NULL,
      result_desc.get(),
      preference,
      requested_algo_count,
      &return_algo_count);

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetMatMulExHeuristicResult(
      matmul_hr.hr(), matmul_algo.mut_algo(), &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

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
        // Not use beta, set beta to 0.0
        beta = 0.0;
        TORCH_CNNL_CHECK(cnnlMatMulEx(
            handle,
            matmul_desc.desc(),
            &alpha,
            mat1_desc.get(),
            mat1_ptr,
            mat2_desc.get(),
            mat2_ptr,
            &beta,
            NULL,
            NULL,
            result_desc.get(),
            result_ptr,
            matmul_algo.algo(),
            workspace_ptr.get(),
            workspace_size));
      });
}

} // namespace ops
} // namespace torch_mlu
