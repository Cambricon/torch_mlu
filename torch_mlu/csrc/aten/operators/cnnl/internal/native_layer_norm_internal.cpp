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

void cnnl_native_layer_norm_internal(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    at::Tensor& mean,
    at::Tensor& rstd,
    double eps,
    int64_t axis) {
  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  tensorDescPtr_t weight_bias_desc = nullptr;
  void* weight_ptr = nullptr;
  if (weight.defined()) {
    auto weight_impl = getMluTensorImpl(weight);
    weight_ptr = mlu_data_ptr(weight_impl);
    weight_bias_desc = getTensorDesc(weight_impl);
  }

  void* bias_ptr = nullptr;
  if (bias.defined()) {
    auto bias_impl = getMluTensorImpl(bias);
    bias_ptr = mlu_data_ptr(bias_impl);
  }

  auto mean_impl = getMluTensorImpl(mean);
  auto mean_rstd_desc = getTensorDesc(mean_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);

  auto rstd_impl = getMluTensorImpl(rstd);
  auto rstd_ptr = mlu_data_ptr(rstd_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // get workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetLayerNormOpWorkspaceSize(
      handle, axis, input_desc.get(), &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlLayerNormForward_v2(
      /*handle        */ handle,
      /*layernorm_desc*/ NULL,
      /*x_desc        */ input_desc.get(),
      /*x             */ input_ptr,
      /*axis          */ axis,
      /*w_b_desc      */ weight_bias_desc.get(),
      /*weight        */ weight_ptr,
      /*bias          */ bias_ptr,
      /*eps           */ eps,
      /*workspace     */ workspace_ptr.get(),
      /*work_size     */ workspace_size,
      /*y_desc        */ output_desc.get(),
      /*y             */ output_ptr,
      /*m_r_desc      */ mean_rstd_desc.get(),
      /*saved_mean    */ mean_ptr,
      /*saved_rstd    */ rstd_ptr));

  return;
}

} // namespace ops
} // namespace torch_mlu
