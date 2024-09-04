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

std::tuple<at::Tensor, at::Tensor> gru_cell_forward_internal(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& input_bias,
    const at::Tensor& hidden_bias,
    const at::Tensor& hx,
    const at::Tensor& hy,
    at::Tensor& workspace) {
  auto handle = getCurrentHandle();
  tensorDescPtr_t input_bias_desc;
  tensorDescPtr_t hidden_bias_desc;
  auto input_impl = getMluTensorImpl(input_gates);
  auto hidden_impl = getMluTensorImpl(hidden_gates);
  auto hx_impl = getMluTensorImpl(hx);
  auto hy_impl = getMluTensorImpl(hy);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto hidden_ptr = hidden_impl->mlu_data_ptr();
  auto hx_ptr = hx_impl->mlu_data_ptr();
  auto hy_ptr = hy_impl->mlu_data_ptr();
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->mlu_data_ptr();
  auto gates_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NGC);
  auto h_desc = getTensorDesc(hy_impl, CNNL_LAYOUT_NC);
  auto workspace_desc = getTensorDesc(workspace_impl, CNNL_LAYOUT_NGC);
  // Get bias ptr when bias is defined.
  void* input_bias_ptr = nullptr;
  void* hidden_bias_ptr = nullptr;
  if (input_bias.defined()) {
    auto input_bias_impl = getMluTensorImpl(input_bias);
    input_bias_ptr = input_bias_impl->mlu_data_ptr();
    input_bias_desc = getTensorDesc(input_bias_impl, CNNL_LAYOUT_NC);
  }
  if (hidden_bias.defined()) {
    auto hidden_bias_impl = getMluTensorImpl(hidden_bias);
    hidden_bias_ptr = hidden_bias_impl->mlu_data_ptr();
    hidden_bias_desc = getTensorDesc(hidden_bias_impl, CNNL_LAYOUT_NC);
  }
  TORCH_CNNL_CHECK(cnnlGRUCellForward(
      /* cnnlHandle_t */ handle,
      /* x_gates_desc */ gates_desc.get(),
      /* x_gates      */ input_ptr,
      /* x_bias_desc  */ input_bias_desc.get(),
      /* x_bias       */ input_bias_ptr,
      /* h_gates_desc */ gates_desc.get(),
      /* h_gates      */ hidden_ptr,
      /* h_bias_desc  */ hidden_bias_desc.get(),
      /* h_bias       */ hidden_bias_ptr,
      /* hx_desc      */ h_desc.get(),
      /* hx           */ hx_ptr,
      /* hy_desc      */ h_desc.get(),
      /* hy           */ hy_ptr,
      /* storage_desc */ workspace_desc.get(),
      /* storage      */ workspace_ptr));
  return std::make_tuple(hy, workspace);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gru_cell_backward_internal(
    const at::Tensor& grad_hy,
    const at::Tensor& workspace,
    const at::Tensor& grad_input_gates,
    const at::Tensor& grad_hidden_gates,
    const at::Tensor& grad_hx) {
  auto handle = getCurrentHandle();
  auto grad_input_gates_impl = getMluTensorImpl(grad_input_gates);
  auto grad_input_gates_ptr = grad_input_gates_impl->mlu_data_ptr();
  auto grad_hidden_gates_impl = getMluTensorImpl(grad_hidden_gates);
  auto grad_hidden_gates_ptr = grad_hidden_gates_impl->mlu_data_ptr();
  auto grad_hx_impl = getMluTensorImpl(grad_hx);
  auto grad_hx_ptr = grad_hx_impl->mlu_data_ptr();
  auto grad_hy_impl = getMluTensorImpl(grad_hy);
  auto grad_hy_ptr = grad_hy_impl->mlu_data_ptr();
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = workspace_impl->mlu_data_ptr();
  auto grad_hx_desc = getTensorDesc(grad_hx_impl, CNNL_LAYOUT_NC);
  auto grad_hy_desc = getTensorDesc(grad_hy_impl, CNNL_LAYOUT_NC);
  auto grad_gates_desc = getTensorDesc(grad_input_gates_impl, CNNL_LAYOUT_NGC);
  auto workspace_desc = getTensorDesc(workspace_impl, CNNL_LAYOUT_NGC);

  TORCH_CNNL_CHECK(cnnlGRUCellBackward(
      /* cnnlHandle_t    */ handle,
      /* grad_hy_desc    */ grad_hy_desc.get(),
      /* grad_hy         */ grad_hy_ptr,
      /* storage_desc    */ workspace_desc.get(),
      /* storage         */ workspace_ptr,
      /* grad_input_desc */ grad_gates_desc.get(),
      /* grad_input      */ grad_input_gates_ptr,
      /* grad_input_desc */ grad_gates_desc.get(),
      /* grad_hidden     */ grad_hidden_gates_ptr,
      /* grad_hx_desc    */ grad_hx_desc.get(),
      /* grad_hx         */ grad_hx_ptr));
  return std::make_tuple(grad_input_gates, grad_hidden_gates, grad_hx);
}

} // namespace ops
} // namespace torch_mlu
