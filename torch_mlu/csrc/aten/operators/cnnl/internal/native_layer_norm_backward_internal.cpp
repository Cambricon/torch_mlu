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
  auto x_impl = getMluTensorImpl(x);
  auto weight_impl = getMluTensorImpl(weight);
  auto diff_x_impl = getMluTensorImpl(diff_x);
  auto diff_weight_impl = getMluTensorImpl(diff_weight);
  auto diff_weight_ptr = diff_weight_impl->mlu_data_ptr();
  auto diff_bias_impl = getMluTensorImpl(diff_bias);
  auto diff_bias_ptr = diff_bias_impl->mlu_data_ptr();
  auto mean_impl = getMluTensorImpl(mean);
  auto rstd_impl = getMluTensorImpl(rstd);

  // get current handle
  auto handle = getCurrentHandle();

  CnnlTensorDescriptor x_desc;
  CnnlTensorDescriptor weight_bias_desc;
  CnnlTensorDescriptor mean_rstd_desc;
  CnnlTensorDescriptor diff_x_desc;
  CnnlTensorDescriptor diff_z_desc;

  // get cnnl descriptor
  x_desc.set(x);
  weight_bias_desc.set(weight);
  if (mean.defined() || rstd.defined()) {
    auto mean_rstd = mean.defined() ? mean : rstd;
    mean_rstd_desc.set(mean_rstd);
  }

  diff_x_desc.set(diff_x);
  diff_z_desc.set(diff_z);

  // malloc mlu memory for input
  auto diff_x_ptr = diff_x_impl->mlu_data_ptr();
  auto x_ptr = x_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto diff_z_ptr = diff_z_impl->mlu_data_ptr();
  void* mean_ptr = mean_impl->mlu_data_ptr();
  void* rstd_ptr = rstd_impl->mlu_data_ptr();

  // get workspace for LNB
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetLayerNormBackwardWorkspaceSize(
      handle, x_desc.desc(), axis, &workspace_size));

  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlLayerNormBackward_v2(
      /*handle     */ handle,
      /*x_desc     */ x_desc.desc(),
      /*x          */ x_ptr,
      /*axis       */ axis,
      /*diff_z_desc*/ diff_z_desc.desc(),
      /*diff_z     */ diff_z_ptr,
      /*w_b_desc   */ weight_bias_desc.desc(),
      /*weight     */ weight_ptr,
      /*m_rstd_desc*/ mean_rstd_desc.desc(),
      /*saved_mean */ mean_ptr,
      /*saved_rstd */ rstd_ptr,
      /*work_space */ ws_ptr.get(),
      /* workspace_size */ workspace_size,
      /*diif_x_desc*/ diff_x_desc.desc(),
      /*diff_x     */ diff_x_ptr,
      /*diff_weight*/ diff_weight_ptr,
      /*diff_bias  */ diff_bias_ptr));

  return;
}

} // namespace ops
} // namespace torch_mlu
