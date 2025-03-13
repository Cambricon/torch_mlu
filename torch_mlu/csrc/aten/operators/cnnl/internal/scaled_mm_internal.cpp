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
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

void cnnl_scaled_mm_out_internal(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    bool is_trans_mat1_,
    bool is_trans_mat2_,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    const at::Tensor& bias) {
  auto mat1_impl = getMluTensorImpl(mat1);
  auto mat1_desc = getTensorDesc(mat1_impl);
  auto mat1_ptr = mat1_impl->mlu_data_ptr();

  auto mat2_impl = getMluTensorImpl(mat2);
  auto mat2_desc = getTensorDesc(mat2_impl);
  auto mat2_ptr = mat2_impl->mlu_data_ptr();

  auto mat1_cnnl_type = getCnnlType(mat1_impl);
  auto mat2_cnnl_type = getCnnlType(mat2_impl);

  auto result_impl = getMluTensorImpl(result);
  auto result_desc = getTensorDesc(result_impl);
  auto result_ptr = result_impl->mlu_data_ptr();

  auto scale_a_impl = getMluTensorImpl(scale_a);
  auto scale_a_ptr = scale_a_impl->mlu_data_ptr();
  auto scale_b_impl = getMluTensorImpl(scale_b);
  auto scale_b_ptr = scale_b_impl->mlu_data_ptr();

  // create desc
  CnnlMatmulExDescriptor matmul_desc;
  CnnlMatmulExAlgorithm matmul_algo;
  cnnlMatMulExPrefer_t preference;
  CnnlMatmulExHeuristicResult matmul_hr;

  int return_algo_count;
  int requested_algo_count = 1;
  int32_t matmul_use_beta = 0;
  int32_t is_trans_mat1 = is_trans_mat1_;
  int32_t is_trans_mat2 = is_trans_mat2_;
  int32_t allow_tf32 = 0;

  CnnlQuantizeExDescriptor quant_desc_a, quant_desc_b;
  quant_desc_a.set(
      nullptr,
      scale_a_ptr,
      nullptr,
      CNNL_POINTER_MODE_DEVICE,
      CNNL_QUANTIZE_PER_TENSOR,
      CNNL_QUANTIZE_SCALE,
      mat1_cnnl_type);

  quant_desc_b.set(
      nullptr,
      scale_b_ptr,
      nullptr,
      CNNL_POINTER_MODE_DEVICE,
      CNNL_QUANTIZE_PER_TENSOR,
      CNNL_QUANTIZE_SCALE,
      mat2_cnnl_type);

  auto compute_cnnl_type = CNNL_DTYPE_FLOAT;

  matmul_desc.set_attr(
      CNNL_MATMUL_EX_DESC_COMPUTE_TYPE,
      &(compute_cnnl_type),
      sizeof(compute_cnnl_type));

  matmul_desc.set_attr(
      CNNL_MATMUL_EX_DESC_TRANSA, &(is_trans_mat1), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_DESC_TRANSB, &(is_trans_mat2), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_USE_BETA, &(matmul_use_beta), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_ALLOW_TF32, &(allow_tf32), sizeof(int32_t));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_A_QUANT, &(quant_desc_a), sizeof(quant_desc_a));
  matmul_desc.set_attr(
      CNNL_MATMUL_EX_B_QUANT, &(quant_desc_b), sizeof(quant_desc_b));

  if (bias.defined()) {
    auto bias_impl = getMluTensorImpl(bias);
    auto bias_desc = getTensorDesc(bias_impl);
    auto bias_ptr = bias_impl->mlu_data_ptr();
    TORCH_CNNL_CHECK(
        cnnlSetMatMulExBias(matmul_desc.desc(), bias_desc.get(), bias_ptr));
  }

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

  const float alpha = 1.0;
  const float beta = 0.0;

  AT_DISPATCH_MLU_FLOAT8(mat1.scalar_type(), "MLU scaled_mm", [&] {
    TORCH_CNNL_CHECK(cnnlMatMulEx_v2(
        handle,
        matmul_desc.desc(),
        matmul_algo.algo(),
        &alpha,
        mat1_desc.get(),
        mat1_ptr,
        mat2_desc.get(),
        mat2_ptr,
        &beta,
        NULL,
        NULL,
        workspace_ptr.get(),
        workspace_size,
        result_desc.get(),
        result_ptr));
  });
}

} // namespace ops
} // namespace torch_mlu
