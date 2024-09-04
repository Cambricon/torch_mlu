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
    at::Tensor& result,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& alpha_,
    const at::Scalar& beta_,
    bool is_trans_batch1_,
    bool is_trans_batch2_,
    bool allow_tf32_) {
  // get tensor impl
  auto batch1_impl = getMluTensorImpl(batch1);
  auto batch2_impl = getMluTensorImpl(batch2);
  auto result_impl = getMluTensorImpl(result);

  // create the desc
  CnnlMatmulDescriptor bmm_desc;
  CnnlMatmulAlgorithm bmm_algo;
  cnnlMatMulPrefer_t preference;
  CnnlBatchMatmulHeuristicResult bmm_hr;

  int return_algo_count;
  int requested_algo_count = 1;
  int32_t use_stride = beta_.to<c10::complex<double>>() == 0.0 ? 1 : 0;
  int32_t is_trans_batch1 = is_trans_batch1_;
  int32_t is_trans_batch2 = is_trans_batch2_;
  int32_t allow_tf32 = allow_tf32_;

  bmm_desc.set_attr(CNNL_MATMUL_ALLOW_TF32, &(allow_tf32), sizeof(int32_t));
  bmm_desc.set_attr(CNNL_MATMUL_USE_STRIDE, &(use_stride), sizeof(int32_t));
  bmm_desc.set_attr(
      CNNL_MATMUL_DESC_TRANSA, &(is_trans_batch1), sizeof(int32_t));
  bmm_desc.set_attr(
      CNNL_MATMUL_DESC_TRANSB, &(is_trans_batch2), sizeof(int32_t));

  auto input_cnnl_type = getCnnlType(batch1_impl);
  auto compute_cnnl_type = input_cnnl_type == CNNL_DTYPE_HALF ||
          input_cnnl_type == CNNL_DTYPE_BFLOAT16
      ? CNNL_DTYPE_FLOAT
      : input_cnnl_type;
  auto batch1_desc =
      getTensorDesc(batch1_impl, input_cnnl_type, CNNL_LAYOUT_ARRAY);
  auto batch2_desc =
      getTensorDesc(batch2_impl, input_cnnl_type, CNNL_LAYOUT_ARRAY);
  auto result_desc = getTensorDesc(
      result_impl, input_cnnl_type, CNNL_LAYOUT_ARRAY, compute_cnnl_type);

  // get current handle
  auto handle = getCurrentHandle();

  bmm_hr.get(
      handle,
      bmm_desc.desc(),
      batch1_desc.get(),
      batch2_desc.get(),
      result_desc.get(),
      preference,
      requested_algo_count,
      &return_algo_count);

  size_t workspace_size = 0;

  TORCH_CNNL_CHECK(cnnlGetBatchMatMulHeuristicResult(
      bmm_hr.hr(), bmm_algo.mut_algo(), &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  // get the mlu ptr
  auto batch1_ptr = batch1_impl->mlu_data_ptr();
  auto batch2_ptr = batch2_impl->mlu_data_ptr();
  auto result_ptr = result_impl->mlu_data_ptr();

  // complex are not supported
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      result.scalar_type(),
      "MLU bmm",
      [&] {
        // More details in Note [beta and alpha type in matmul ops]
        using math_type = MLUAccumulateType_t<scalar_t>;
        auto alpha = alpha_.to<math_type>();
        auto beta = beta_.to<math_type>();
        TORCH_CNNL_CHECK(cnnlBatchMatMulBCast_v2(
            handle,
            bmm_desc.desc(),
            bmm_algo.algo(),
            &alpha,
            batch1_desc.get(),
            batch1_ptr,
            batch2_desc.get(),
            batch2_ptr,
            &beta,
            result_desc.get(),
            result_ptr,
            workspace_ptr.get(),
            workspace_size));
      });
}
} // namespace ops
} // namespace torch_mlu
