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

using at::Tensor;
using at::TensorList;

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>
cnnl_rnn_backward_internal(
    const Tensor& input_r,
    const TensorList& weight,
    const Tensor& weight_buf,
    int64_t weight_stride0,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output_r,
    const Tensor& grad_output_r,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t proj_size,
    int64_t fn_num_layers,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    const int* batch_sizes_int_ptr,
    const at::Tensor& dev_batch_sizes,
    const Tensor& fn_dropout_state,
    const Tensor& fn_reserve,
    std::array<bool, 3> output_mask) {
  TORCH_CHECK(hx.is_contiguous(), "cnnl rnn backward: hx is not contiguous");
  TORCH_CHECK(
      !cx.defined() || cx.is_contiguous(),
      "cnnl rnn backward: cx is not contiguous");
  TORCH_CHECK(
      (cnnlRNNMode_t)fn_mode == CNNL_LSTM && cx.defined(),
      "cnnl rnn backward currently only support LSTM RNN and cx must be defined.");

  auto input = input_r;
  auto handle = getCurrentHandle();
  auto input_type = getCnnlDataType(input.dtype());
  auto grad_output = grad_output_r;
  auto output = output_r;
  auto input_size = dev_batch_sizes.defined() ? input.size(1) : input.size(2);

  auto dx = at::empty(input.sizes(), input.options());
  auto dhy = cnnl_contiguous(grad_hy).view(hx.sizes());
  auto dcy =
      grad_cy.defined() ? cnnl_contiguous(grad_cy).view(cx.sizes()) : Tensor();
  auto dhx = at::empty(hx.sizes(), hx.options());
  auto dcx = cx.defined() ? at::empty(cx.sizes(), cx.options()) : Tensor();
  auto dw = at::zeros(weight_buf.sizes(), weight_buf.options());

  TORCH_CHECK(
      fn_train, "cnnl RNN backward can only be called in training mode");

  TORCH_CHECK(
      dhy.device().is_privateuseone() &&
          grad_output.device().is_privateuseone() &&
          (!dcy.defined() || dcy.device().is_privateuseone()),
      "Gradients aren't MLU tensors");

  // Get seq_arr and max_batch_size.
  const int seq_arr_size = dev_batch_sizes.defined()
      ? c10::checked_convert<int32_t, size_t>(
            dev_batch_sizes.numel(), "int32_t")
      : 0;

  // input
  CnnlSeqDataDescriptor input_desc;
  input_desc.set(input, seq_arr_size, batch_sizes_int_ptr);
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->mlu_data_ptr();

  // output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  CnnlSeqDataDescriptor output_desc;
  output_desc.set(output, seq_arr_size, batch_sizes_int_ptr);

  // grad_output
  auto dy_impl = getMluTensorImpl(grad_output);
  auto dy_ptr = dy_impl->mlu_data_ptr();

  // grad_dw
  auto* dw_impl = getMluTensorImpl(dw);
  auto dw_ptr = dw_impl->mlu_data_ptr();

  // RNNDesc
  if (proj_size != 0) {
    --weight_stride0;
  }
  bool has_biases = weight_stride0 == 4 ? true : false;
  CnnlRNNDescriptor rnn_desc;
  rnn_desc.set(
      c10::checked_convert<int32_t, int64_t>(fn_hidden_size, "int32_t"),
      c10::checked_convert<int32_t, int64_t>(proj_size, "int32_t"),
      c10::checked_convert<int32_t, int64_t>(fn_num_layers, "int32_t"),
      c10::checked_convert<int32_t, int64_t>(input_size, "int32_t"),
      has_biases,
      fn_bidirectional,
      (cnnlRNNMode_t)fn_mode,
      input_type);

  // dx
  auto dx_impl = getMluTensorImpl(dx);
  auto dx_ptr = dx_impl->mlu_data_ptr();

  // hy, cy
  auto hx_impl = getMluTensorImpl(hx);
  auto hx_ptr = hx_impl->mlu_data_ptr();
  auto cx_impl = getMluTensorImpl(cx);
  auto cx_ptr = cx_impl->mlu_data_ptr();
  auto hx_desc = getTensorDesc(hx_impl);
  auto cx_desc = getTensorDesc(cx_impl);

  // dhy, dcy
  auto dhy_impl = getMluTensorImpl(dhy);
  auto dhy_ptr = dhy_impl->mlu_data_ptr();
  auto dcy_impl = getMluTensorImpl(dcy);
  auto dcy_ptr = dcy_impl->mlu_data_ptr();

  // dhx, dcx
  auto dhx_impl = getMluTensorImpl(dhx);
  auto dhx_ptr = dhx_impl->mlu_data_ptr();
  auto dcx_impl = getMluTensorImpl(dcx);
  auto dcx_ptr = dcx_impl->mlu_data_ptr();

  size_t reserve_size = 0;
  size_t workspace_size = 0;
  cnnlGetRNNTempSizes(
      handle,
      rnn_desc.desc(),
      input_desc.desc(),
      &workspace_size,
      &reserve_size);
  // RNNWorkspaceSize
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // reserve
  auto reserve_impl = getMluTensorImpl(fn_reserve);
  auto reserve_ptr = reserve_impl->mlu_data_ptr();

  // weight
  auto* weightspace_impl = getMluTensorImpl(weight_buf);
  auto weightspace_ptr = weightspace_impl->mlu_data_ptr();
  // onchip dtype is float when the dtype of tensor is double.
  auto weight_nbytes = weight_buf.dtype() == at::kDouble
      ? weight_buf.nbytes() / 2
      : weight_buf.nbytes();

  int* dev_seq_lengths_ptr = dev_batch_sizes.defined()
      ? static_cast<int*>(dev_batch_sizes.data_ptr())
      : nullptr;

  // RNNBackwardData
  TORCH_CNNL_CHECK(cnnlRNNBackward(
      handle,
      rnn_desc.desc(),
      dev_seq_lengths_ptr,
      output_desc.desc(),
      output_ptr,
      dy_ptr,
      input_desc.desc(),
      input_ptr,
      dx_ptr,
      hx_desc.get(),
      hx_ptr,
      dhy_ptr,
      dhx_ptr,
      cx_desc.get(),
      cx_ptr,
      dcy_ptr,
      dcx_ptr,
      weightspace_ptr,
      dw_ptr,
      weight_nbytes,
      workspace_ptr.get(),
      workspace_size,
      reserve_ptr,
      reserve_ptr,
      fn_reserve.nbytes()));

  // split weight gradient
  int64_t index = 0;
  std::vector<at::Tensor> dweight_vec;
  if (has_biases && proj_size > 0) {
    // Different with cudnn which use cudnnGetRNNLinLayerMatrixParams to calc
    // the position of w_hr, cnnl use the fixed order to try to align with
    // cudnn, w_hr is infront of bias.
    dweight_vec.resize(weight.size());
    for (auto& i : {0, 1, 4, 2, 3}) {
      dweight_vec[i] =
          dw.slice(0, index, index + weight[i].numel()).view(weight[i].sizes());
      index += weight[i].numel();
    }
    if (fn_bidirectional) {
      for (auto& i : {5, 6, 9, 7, 8}) {
        dweight_vec[i] = dw.slice(0, index, index + weight[i].numel())
                             .view(weight[i].sizes());
        index += weight[i].numel();
      }
    }
  } else {
    for (auto& w : weight) {
      auto w_buf = dw.slice(0, index, index + w.numel());
      dweight_vec.emplace_back(w_buf.view(w.sizes()));
      index += w.numel();
    }
  }

  return std::make_tuple(dx, dhx, dcx, dweight_vec);
}

} // namespace ops
} // namespace torch_mlu
