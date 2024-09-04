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

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_cell_forward_internal(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& input_bias,
    const at::Tensor& hidden_bias,
    const at::Tensor& cx,
    const at::Tensor& hy,
    const at::Tensor& cy,
    at::Tensor& workspace) {
  auto handle = getCurrentHandle();
  auto input_impl = getMluTensorImpl(input_gates);
  auto hidden_impl = getMluTensorImpl(hidden_gates);
  auto cx_impl = getMluTensorImpl(cx);
  auto hy_impl = getMluTensorImpl(hy);
  auto cy_impl = getMluTensorImpl(cy);
  auto gates_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NGC);
  auto c_desc = getTensorDesc(cx_impl, CNNL_LAYOUT_NC);
  auto h_desc = getTensorDesc(hy_impl, CNNL_LAYOUT_NC);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto hidden_ptr = mlu_data_ptr(hidden_impl);
  auto cx_ptr = mlu_data_ptr(cx_impl);
  auto hy_ptr = mlu_data_ptr(hy_impl);
  auto cy_ptr = mlu_data_ptr(cy_impl);
  // Get workspace size, and never be zero.
  size_t reserve_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetLSTMGatesTempSize(handle, gates_desc.get(), &reserve_size));
  workspace = at::empty(
      {static_cast<int64_t>(reserve_size)},
      input_gates.options().dtype(at::kByte));
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = mlu_data_ptr(workspace_impl);
  // Get bias ptr when bias is defined.
  auto get_tensor_ptr = [](const at::Tensor& t) -> void* {
    void* ptr = nullptr;
    if (t.defined()) {
      auto t_impl = getMluTensorImpl(t);
      ptr = mlu_data_ptr(t_impl);
    }
    return ptr;
  };
  void* input_bias_ptr = get_tensor_ptr(input_bias);
  void* hidden_bias_ptr = get_tensor_ptr(hidden_bias);
  TORCH_CNNL_CHECK(cnnlLSTMGatesForward(
      /* cnnlHandle_t */ handle,
      /* gates_desc   */ gates_desc.get(),
      /* x_gates      */ input_ptr,
      /* x_bias       */ input_bias_ptr,
      /* h_gates      */ hidden_ptr,
      /* h_bias       */ hidden_bias_ptr,
      /* c_desc       */ c_desc.get(),
      /* cx           */ cx_ptr,
      /* cy           */ cy_ptr,
      /* h_desc       */ h_desc.get(),
      /* hy           */ hy_ptr,
      /* reservespace */ workspace_ptr,
      /* reserve_size */ reserve_size));
  return std::make_tuple(hy, cy, workspace);
}

std::tuple<at::Tensor, at::Tensor> lstm_cell_backward_internal(
    const at::Tensor& grad_hy,
    const at::Tensor& grad_cy,
    const at::Tensor& cx,
    const at::Tensor& cy,
    const at::Tensor& workspace,
    const at::Tensor& grad_gates,
    const at::Tensor& grad_cx) {
  auto handle = getCurrentHandle();
  const at::Tensor& defined_grad = grad_hy.defined() ? grad_hy : grad_cy;
  auto defined_grad_impl = getMluTensorImpl(defined_grad);
  auto grad_gates_impl = getMluTensorImpl(grad_gates);
  auto grad_gates_ptr = mlu_data_ptr(grad_gates_impl);
  // cx, cy, grad_cx
  auto cx_impl = getMluTensorImpl(cx);
  auto cy_impl = getMluTensorImpl(cy);
  auto grad_cx_impl = getMluTensorImpl(grad_cx);
  auto grad_hy_desc = getTensorDesc(defined_grad_impl, CNNL_LAYOUT_NC);
  auto cx_desc = getTensorDesc(cx_impl, CNNL_LAYOUT_NC);
  auto grad_gates_desc = getTensorDesc(grad_gates_impl, CNNL_LAYOUT_NGC);
  auto cx_ptr = mlu_data_ptr(cx_impl);
  auto cy_ptr = mlu_data_ptr(cy_impl);
  auto grad_cx_ptr = mlu_data_ptr(grad_cx_impl);
  auto workspace_impl = getMluTensorImpl(workspace);
  auto workspace_ptr = mlu_data_ptr(workspace_impl);
  // Get grad_hy / grad_cy ptr when bias is defined.
  auto get_tensor_ptr = [](const at::Tensor& t) -> void* {
    void* ptr = nullptr;
    if (t.defined()) {
      auto t_impl = getMluTensorImpl(t);
      ptr = mlu_data_ptr(t_impl);
    }
    return ptr;
  };
  void* grad_hy_ptr = get_tensor_ptr(grad_hy);
  void* grad_cy_ptr = get_tensor_ptr(grad_cy);
  TORCH_CNNL_CHECK(cnnlLSTMGatesBackward(
      /* cnnlHandle_t      */ handle,
      /* grad_hy_desc      */ grad_hy_desc.get(),
      /* grad_hy           */ grad_hy_ptr,
      /* cx_desc           */ cx_desc.get(),
      /* cx                */ cx_ptr,
      /* cy                */ cy_ptr,
      /* grad_cy           */ grad_cy_ptr,
      /* grad_cx           */ grad_cx_ptr,
      /* grad_gates_desc   */ grad_gates_desc.get(),
      /* grad_gates        */ grad_gates_ptr,
      /* reservespace      */ workspace_ptr,
      /* reservespace_size */ workspace.nbytes()));
  return std::make_tuple(grad_gates, grad_cx);
}

} // namespace ops
} // namespace torch_mlu
