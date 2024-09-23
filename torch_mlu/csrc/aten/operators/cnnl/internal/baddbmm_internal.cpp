/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All batch2 contributions:
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
      documentation and/or batch2 materials provided with the distribution.
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
OR TORT (INCLUDING NEGLIGENCE OR batch2WISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_baddbmm_out_internal(
    bool transa,
    bool transb,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t batch_size,
    const at::Tensor& self,
    int32_t ldc,
    int64_t stride_c,
    const at::Scalar& alpha,
    const at::Tensor& batch1,
    int32_t lda,
    int64_t stride_a,
    const at::Tensor& batch2,
    int32_t ldb,
    int64_t stride_b,
    const at::Scalar& beta,
    at::Tensor& result,
    int32_t ldd,
    int64_t stride_d,
    bool allow_tf32_) {
  // get tensor impl
  auto batch1_impl = getMluTensorImpl(batch1);
  auto batch2_impl = getMluTensorImpl(batch2);
  auto result_impl = getMluTensorImpl(result);
  auto self_impl = getMluTensorImpl(self);

  // create the desc
  CnnlStrideBatchMatmulDescriptor bmm_desc;
  CnnlStrideBatchMatmulAlgorithm bmm_algo;
  cnnlMatMulPrefer_t preference;
  CnnlStrideBatchMatmulHeuristicResult bmm_hr;

  int return_algo_count;
  int requested_algo_count = 1;
  int32_t use_stride = beta.to<c10::complex<double>>() == 0.0 ? 1 : 0;
  int32_t is_trans_batch1 = int(transa);
  int32_t is_trans_batch2 = int(transb);
  int32_t allow_tf32 = allow_tf32_;

  bmm_desc.set_attr(CNNL_STRIDE_BMM_ALLOW_TF32, &(allow_tf32), sizeof(int32_t));

  auto input_cnnl_type = getCnnlType(batch1_impl);
  auto compute_cnnl_type = input_cnnl_type == CNNL_DTYPE_HALF ||
          input_cnnl_type == CNNL_DTYPE_BFLOAT16
      ? CNNL_DTYPE_FLOAT
      : input_cnnl_type;
  auto self_desc = getTensorDesc(self_impl, input_cnnl_type, CNNL_LAYOUT_ARRAY);
  auto batch1_desc =
      getTensorDesc(batch1_impl, input_cnnl_type, CNNL_LAYOUT_ARRAY);
  auto batch2_desc =
      getTensorDesc(batch2_impl, input_cnnl_type, CNNL_LAYOUT_ARRAY);
  auto result_desc = getTensorDesc(
      result_impl, input_cnnl_type, CNNL_LAYOUT_ARRAY, compute_cnnl_type);

  int batch_size_array = {batch_size};
  auto alpha_ = alpha.to<float>();
  auto beta_ = beta.to<float>();
  // get current handle
  auto handle = getCurrentHandle();

  bmm_hr.get(
      handle,
      transa,
      transb,
      alpha_,
      beta_,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      ldd,
      stride_a,
      stride_b,
      stride_c,
      stride_d,
      &batch_size_array,
      bmm_desc.desc(),
      self_desc.get(),
      batch1_desc.get(),
      batch2_desc.get(),
      result_desc.get(),
      preference,
      requested_algo_count,
      &return_algo_count);

  size_t workspace_size = 0;
  cnnlStrideBatchMatMulAlgo_t algo_ = bmm_algo.mut_algo();
  TORCH_CNNL_CHECK(cnnlGetStrideBatchMatMulHeuristicResult(
      bmm_hr.hr(), &algo_, &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  // get the mlu ptr
  auto batch1_ptr = mlu_data_ptr(batch1_impl);
  auto batch2_ptr = mlu_data_ptr(batch2_impl);
  auto result_ptr = mlu_data_ptr(result_impl);
  auto self_ptr = mlu_data_ptr(self_impl);

  // complex are not supported
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      result.scalar_type(),
      "MLU bmm",
      [&] {
        TORCH_CNNL_CHECK(cnnlStrideBatchMatMul_v3(
            handle,
            bmm_desc.desc(),
            bmm_algo.algo(),
            transa,
            transb,
            m,
            n,
            k,
            &batch_size_array,
            &alpha_,
            batch1_desc.get(),
            batch1_ptr,
            lda,
            &stride_a,
            batch2_desc.get(),
            batch2_ptr,
            ldb,
            &stride_b,
            &beta_,
            self_desc.get(),
            self_ptr,
            ldc,
            &stride_c,
            workspace_ptr.get(),
            workspace_size,
            result_desc.get(),
            result_ptr,
            ldd,
            &stride_d));
      });
}
} // namespace ops
} // namespace torch_mlu
