/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
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

// Currently only support CNNL_LAYOUT_NC, CNNL_LAYOUT_NHWC and CNNL_LAYOUT_NDHWC
// layouts of input, and output must be the same layout with input.
void cnnl_native_batch_norm_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& scale_t,
    const at::Tensor& bias_t,
    const at::Tensor& moving_mean,
    const at::Tensor& moving_var,
    Tensor& saved_mean,
    Tensor& saved_invstd,
    bool training,
    double momentum,
    double eps) {
  int64_t c_dim = input.size(1);
  auto scale = scale_t;
  auto bias = bias_t;
  // TODO(PYTORCH-9290): Can be deleted when CNNL support nullptr.
  if (!scale.defined() && training) {
    scale = at::empty({c_dim}, saved_mean.options()).fill_(1);
  }
  if (!bias.defined() && training) {
    bias = at::empty({c_dim}, saved_mean.options()).fill_(0);
  }

  // get current handle
  auto handle = getCurrentHandle();

  // set cnnl descriptor
  auto dim = input.dim();
  auto layout = dim == 2 ? CNNL_LAYOUT_NC
                         : (dim > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC);

  // malloc mlu memory
  auto* input_impl = getMluTensorImpl(input);
  auto* output_impl = getMluTensorImpl(output);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  void* scale_ptr =
      scale.defined() ? getMluTensorImpl(scale)->mlu_data_ptr() : nullptr;
  void* bias_ptr =
      bias.defined() ? getMluTensorImpl(bias)->mlu_data_ptr() : nullptr;

  void* moving_mean_ptr = moving_mean.defined()
      ? getMluTensorImpl(moving_mean)->mlu_data_ptr()
      : nullptr;
  void* moving_var_ptr = moving_var.defined()
      ? getMluTensorImpl(moving_var)->mlu_data_ptr()
      : nullptr;

  auto* saved_mean_impl = getMluTensorImpl(saved_mean);
  auto saved_mean_ptr = saved_mean_impl->mlu_data_ptr();
  auto scale_bias_mean_var_desc =
      getTensorDesc(saved_mean_impl, CNNL_LAYOUT_ARRAY);
  auto saved_invstd_ptr = getMluTensorImpl(saved_invstd)->mlu_data_ptr();

  const void* alpha = nullptr;
  const void* beta = nullptr;

  // get workspace for BNFT
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetBatchNormForwardWorkspaceSize(
      handle, input_desc.get(), &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // set activition part to default
  cnnlBatchNormMode_t mode = CNNL_BATCHNORM_SPATIAL;
  cnnlBatchNormOps_t bnOps = CNNL_BATCHNORM_OPS_BN;
  cnnlActivationMode_t active_mode = CNNL_ACTIVATION_IDENTITY;

  if (training) {
    cnnlActivationDescriptor_t activation_desc = nullptr;
    cnnlCreateActivationDescriptor(&activation_desc);
    cnnlSetActivationDescriptor_v5(
        activation_desc,
        active_mode,
        CNNL_ACTIVATION_HIGH_PRECISION,
        CNNL_NOT_PROPAGATE_NAN,
        1.0,
        -1,
        1.0,
        1.0,
        false);
    TORCH_CNNL_CHECK(cnnlBatchNormForwardTraining_v2(
        /* handle            */ handle,
        /* activation_desc   */ activation_desc,
        /* mode              */ mode,
        /* bnOps             */ bnOps,
        /* alpha             */ alpha,
        /* beta              */ beta,
        /* x_desc            */ input_desc.get(),
        /* x                 */ input_ptr,
        /* z_desc            */ NULL,
        /* z                 */ NULL,
        /* wbmvd             */ scale_bias_mean_var_desc.get(),
        /* weight            */ scale_ptr,
        /* bias              */ bias_ptr,
        /* mov_mean          */ moving_mean_ptr,
        /* mov_var           */ moving_var_ptr,
        /* eps               */
        c10::checked_convert<float, double>(eps, "float"),
        /* momentum          */
        c10::checked_convert<float, double>(momentum, "float"),
        /* y_desc            */ output_desc.get(),
        /* y                 */ output_ptr,
        /* save_mean         */ saved_mean_ptr,
        /* save_std          */ saved_invstd_ptr,
        /* workspace         */ ws_ptr.get(),
        /* workspace_size    */ workspace_size,
        /* reservespace      */ NULL,
        /* reservespace_size */ 0));
    // release activation descriptor
    cnnlDestroyActivationDescriptor(activation_desc);
  } else {
    // TODO(PYTORCH-9290): We can not calc save_invstd due to lack of kernel.
    TORCH_CNNL_CHECK(cnnlBatchNormForwardInference(
        /* handle   */ handle,
        /* alpha    */ alpha,
        /* beta     */ beta,
        /* x_desc   */ input_desc.get(),
        /* x        */ input_ptr,
        /* wbmvd    */ scale_bias_mean_var_desc.get(),
        /* weight   */ scale_ptr,
        /* bias     */ bias_ptr,
        /* mov_mean */ moving_mean_ptr,
        /* mov_var  */ moving_var_ptr,
        /* eps      */ c10::checked_convert<float, double>(eps, "float"),
        /* z_desc   */ output_desc.get(),
        /* z        */ output_ptr));
  }
  return;
}

} // namespace ops
} // namespace torch_mlu
