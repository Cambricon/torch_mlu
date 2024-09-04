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
std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_native_batch_norm_backward_internal(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool training,
    double eps) {
  int64_t c_dim = input.size(1);
  auto weight_t = weight;
  // TODO(): Can be deleted when CNNL support nullptr.
  if (!weight_t.defined()) {
    weight_t = at::empty({c_dim}, save_mean.options()).fill_(1);
  }
  at::Tensor diff_weight = at::empty({c_dim}, weight_t.options());
  at::Tensor diff_bias = at::empty({c_dim}, weight_t.options());
  at::Tensor diff_x = at::empty_like(input);

  auto input_impl = getMluTensorImpl(input);
  auto grad_impl = getMluTensorImpl(grad_out);
  auto diff_x_impl = getMluTensorImpl(diff_x);
  auto weight_impl = getMluTensorImpl(weight_t);
  auto mean_impl = getMluTensorImpl(save_mean);
  auto var_impl = getMluTensorImpl(save_invstd);
  auto diff_weight_impl = getMluTensorImpl(diff_weight);
  auto diff_bias_impl = getMluTensorImpl(diff_bias);
  // get current handle
  auto handle = getCurrentHandle();

  // set cnnl descriptor
  auto dim = input.dim();
  auto layout = dim == 2 ? CNNL_LAYOUT_NC
                         : (dim > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto grad_desc = getTensorDesc(grad_impl, layout);
  auto diff_x_desc = getTensorDesc(diff_x_impl, layout);
  auto weight_bias_mean_var_desc = getTensorDesc(mean_impl, CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto grad_ptr = grad_impl->mlu_data_ptr();
  auto diff_x_ptr = diff_x_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto mean_ptr = mean_impl->mlu_data_ptr();
  auto var_ptr = var_impl->mlu_data_ptr();
  auto diff_weight_ptr = diff_weight_impl->mlu_data_ptr();
  auto diff_bias_ptr = diff_bias_impl->mlu_data_ptr();

  const void* alpha_data_diff = nullptr;
  const void* beta_data_diff = nullptr;
  const void* alpha_param_diff = nullptr;
  const void* beta_param_diff = nullptr;

  // set activition part
  cnnlBatchNormMode_t mode = CNNL_BATCHNORM_SPATIAL;
  cnnlBatchNormOps_t bnOps = CNNL_BATCHNORM_OPS_BN;
  cnnlActivationMode_t active_mode = CNNL_ACTIVATION_IDENTITY;

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

  if (training) {
    // get workspace for SYBNReduce
    size_t workspace_size = 0;
    TORCH_CNNL_CHECK(cnnlGetBatchNormBackwardWorkspaceSize(
        handle, input_desc.get(), &workspace_size));

    auto ws_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

    TORCH_CNNL_CHECK(cnnlBatchNormBackward_v2(
        /* handle                     */ handle,
        /* cnnlActivationDescriptor_t */ activation_desc,
        /* cnnlBatchNormMode_t        */ mode,
        /* cnnlBatchNormOps_t         */ bnOps,
        /* alpha_data_diff            */ alpha_data_diff,
        /* beta_data_diff             */ beta_data_diff,
        /* alpha_param_diff           */ alpha_param_diff,
        /* beta_param_diff            */ beta_param_diff,
        /* x_desc                     */ input_desc.get(),
        /* x                          */ input_ptr,
        /* y_desc                     */ NULL,
        /* y                          */ NULL,
        /* diff_y_desc                */ grad_desc.get(),
        /* diff_y                     */ grad_ptr,
        /* filter_bias_mean_var_desc  */ weight_bias_mean_var_desc.get(),
        /* filter                     */ weight_ptr,
        /* bias                       */ NULL,
        /* saved_mean                 */ mean_ptr,
        /* saved_invstd               */ var_ptr,
        /* eps                        */
        c10::checked_convert<float, double>(eps, "float"),
        /* diff_z_desc                */ NULL,
        /* diff_z                     */ NULL,
        /* diff_x_desc                */ diff_x_desc.get(),
        /* diff_x                     */ diff_x_ptr,
        /* diff_filter                */ diff_weight_ptr,
        /* diff_bias                  */ diff_bias_ptr,
        /* workspace                  */ ws_ptr.get(),
        /* workspace_size             */ workspace_size,
        /* reservespace               */ NULL,
        /* reservespace_size          */ 0));
  } else {
    auto running_mean_impl = getMluTensorImpl(running_mean);
    auto running_var_impl = getMluTensorImpl(running_var);
    auto running_mean_ptr = running_mean_impl->mlu_data_ptr();
    auto running_var_ptr = running_var_impl->mlu_data_ptr();

    // get workspace for FrozenBNBackward
    size_t workspace_size = 0;
    TORCH_CNNL_CHECK(cnnlGetFrozenBatchNormBackwardWorkspaceSize(
        handle, input_desc.get(), &workspace_size));
    auto ws_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

    TORCH_CNNL_CHECK(cnnlFrozenBatchNormBackward_v2(
        /* handle                     */ handle,
        /* cnnlActivationDescriptor_t */ activation_desc,
        /* cnnlBatchNormMode_t        */ mode,
        /* cnnlBatchNormOps_t         */ bnOps,
        /* x_desc                     */ input_desc.get(),
        /* x                          */ input_ptr,
        /* y_desc                     */ NULL,
        /* y                          */ NULL,
        /* diff_y_desc                */ grad_desc.get(),
        /* diff_y                     */ grad_ptr,
        /* filter_bias_mean_var_desc  */ weight_bias_mean_var_desc.get(),
        /* filter                     */ weight_ptr,
        /* bias                       */ NULL,
        /* pop_mean                   */ running_mean_ptr,
        /* pop_var                    */ running_var_ptr,
        /* eps                        */
        c10::checked_convert<float, double>(eps, "float"),
        /* workspace                  */ ws_ptr.get(),
        /* workspace_size             */ workspace_size,
        /* diff_z_desc                */ NULL,
        /* diff_z                     */ NULL,
        /* diff_x_desc                */ diff_x_desc.get(),
        /* diff_x                     */ diff_x_ptr,
        /* diff_filter                */ diff_weight_ptr,
        /* diff_bias                  */ diff_bias_ptr));
  }
  // release activation descriptor
  cnnlDestroyActivationDescriptor(activation_desc);
  return std::make_tuple(diff_x, diff_weight, diff_bias);
}

} // namespace ops
} // namespace torch_mlu
