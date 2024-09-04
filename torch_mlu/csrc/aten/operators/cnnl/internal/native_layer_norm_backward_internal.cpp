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
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_native_layer_norm_backward_internal(
    const at::Tensor& diff_z,
    const at::Tensor& x,
    int64_t axis,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& weight,
    at::Tensor& diff_x,
    at::Tensor& diff_weight,
    at::Tensor& diff_bias) {
  auto diff_z_impl = getMluTensorImpl(diff_z);
  auto diff_z_desc = getTensorDesc(diff_z_impl);
  auto diff_z_ptr = mlu_data_ptr(diff_z_impl);

  auto x_impl = getMluTensorImpl(x);
  auto x_desc = getTensorDesc(x_impl);
  auto x_ptr = mlu_data_ptr(x_impl);

  auto weight_impl = getMluTensorImpl(weight);
  auto weight_bias_desc = getTensorDesc(weight_impl);
  auto weight_ptr = mlu_data_ptr(weight_impl);

  auto diff_x_impl = getMluTensorImpl(diff_x);
  auto diff_x_desc = getTensorDesc(diff_x_impl);
  auto diff_x_ptr = mlu_data_ptr(diff_x_impl);

  auto diff_weight_impl = getMluTensorImpl(diff_weight);
  auto diff_weight_ptr = mlu_data_ptr(diff_weight_impl);

  auto diff_bias_impl = getMluTensorImpl(diff_bias);
  auto diff_bias_ptr = mlu_data_ptr(diff_bias_impl);

  auto mean_impl = getMluTensorImpl(mean);
  void* mean_ptr = mlu_data_ptr(mean_impl);

  auto rstd_impl = getMluTensorImpl(rstd);
  void* rstd_ptr = mlu_data_ptr(rstd_impl);

  tensorDescPtr_t mean_rstd_desc = nullptr;
  if (mean.defined() || rstd.defined()) {
    auto mean_rstd = mean.defined() ? mean : rstd;
    auto mean_rstd_impl = getMluTensorImpl(mean_rstd);
    mean_rstd_desc = getTensorDesc(mean_rstd_impl);
  }

  // get current handle
  auto handle = getCurrentHandle();

  // get workspace for LNB
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetLayerNormBackwardWorkspaceSize(
      handle, x_desc.get(), axis, &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlLayerNormBackward_v2(
      /*handle     */ handle,
      /*x_desc     */ x_desc.get(),
      /*x          */ x_ptr,
      /*axis       */ axis,
      /*diff_z_desc*/ diff_z_desc.get(),
      /*diff_z     */ diff_z_ptr,
      /*w_b_desc   */ weight_bias_desc.get(),
      /*weight     */ weight_ptr,
      /*m_rstd_desc*/ mean_rstd_desc.get(),
      /*saved_mean */ mean_ptr,
      /*saved_rstd */ rstd_ptr,
      /*work_space */ ws_ptr.get(),
      /* workspace_size */ workspace_size,
      /*diif_x_desc*/ diff_x_desc.get(),
      /*diff_x     */ diff_x_ptr,
      /*diff_weight*/ diff_weight_ptr,
      /*diff_bias  */ diff_bias_ptr));

  return;
}

} // namespace ops
} // namespace torch_mlu
