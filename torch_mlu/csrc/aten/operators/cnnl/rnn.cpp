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

#include <ATen/native/RNN.h>
#include <ATen/native/Resize.h>
#include <ATen/autocast_mode.h>
#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

using at::Tensor;
using at::TensorList;
using at::native::check_attributes;
using at::native::lstm_cudnn_stub;
using at::native::lstm_packed_cudnn_stub;

namespace torch_mlu {
namespace ops {

////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// LSTM CELL
///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

// Factor will be 3 for GRU and 4 for LSTM
void checkSizesAndDtypes(
    at::CheckedFrom c,
    const at::TensorArg& input_gates,
    const at::TensorArg& hidden_gates,
    const at::TensorArg& input_bias,
    const at::TensorArg& hidden_bias,
    int64_t factor,
    const at::TensorArg& prev_hidden) {
  at::checkDim(c, input_gates, 2);
  at::checkSameSize(c, input_gates, hidden_gates);
  int64_t gates_size = input_gates->size(1);

  if (input_bias->defined()) {
    at::checkDim(c, input_bias, 1);
    at::checkNumel(c, input_bias, gates_size);
    at::checkSameSize(c, input_bias, hidden_bias);
  }

  at::checkDim(c, prev_hidden, 2);
  at::checkNumel(c, prev_hidden, input_gates->size(0) * gates_size / factor);

  checkAllSameMLU(
      c, {input_gates, hidden_gates, input_bias, hidden_bias, prev_hidden});
  at::checkAllSameType(
      c, {input_gates, hidden_gates, input_bias, hidden_bias, prev_hidden});
}

void checkLSTMBackwardSizesAndDtypes(
    const at::TensorArg& grad_hy,
    const at::TensorArg& grad_cy,
    const at::TensorArg& cx,
    const at::TensorArg& cy) {
  at::CheckedFrom c = "cnnl__thnn_fused_lstm_cell_backward_impl";
  const at::TensorArg& defined_grad = grad_hy->defined() ? grad_hy : grad_cy;
  at::checkDim(c, defined_grad, 2);
  auto exp_size = defined_grad->sizes();
  if (grad_hy->defined()) {
    at::checkSize(c, grad_hy, exp_size);
  }
  if (grad_cy->defined()) {
    at::checkSize(c, grad_cy, exp_size);
  }
  at::checkSize(c, cx, exp_size);
  at::checkSize(c, cy, exp_size);
  at::checkAllSameType(c, {grad_hy, grad_cy, cx, cy});
}

std::tuple<Tensor, Tensor, Tensor> cnnl__thnn_fused_lstm_cell(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& cx,
    const c10::optional<at::Tensor>& input_bias_opt,
    const c10::optional<at::Tensor>& hidden_bias_opt) {
  const int gate_num = 4;
  const auto batch = input_gates.size(0);
  const auto hidden_size = input_gates.size(1) / gate_num;
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> input_bias_maybe_owned =
      at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias =
      c10::value_or_else(hidden_bias_opt, [] { return Tensor(); });
  checkSizesAndDtypes(
      "cnnl__thnn_fused_lstm_cell",
      {input_gates, "input_gates", 1},
      {hidden_gates, "hidden_gates", 2},
      {input_bias, "input_bias", 3},
      {hidden_bias, "hidden_bias", 4},
      gate_num,
      {cx, "prev_hidden", 5});
  // reshape Tensor size from [N, gate_num*hidden_size] to [N, gate_num,
  // hidden_size]
  std::vector<int64_t> cnnl_gates_shape = {batch, gate_num, hidden_size};
  auto input_gates_contiguous =
      cnnl_contiguous(input_gates.reshape(cnnl_gates_shape));
  auto hidden_gates_contiguous =
      cnnl_contiguous(hidden_gates.reshape(cnnl_gates_shape));

  auto cx_contiguous = cnnl_contiguous(cx);
  auto input_bias_contiguous =
      input_bias.defined() ? cnnl_contiguous(input_bias) : input_bias;
  auto hidden_bias_contiguous =
      hidden_bias.defined() ? cnnl_contiguous(hidden_bias) : hidden_bias;
  auto hy = at::empty_like(cx_contiguous, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto cy = at::empty_like(cx_contiguous, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::Tensor workspace;
  return lstm_cell_forward_internal(
      input_gates_contiguous,
      hidden_gates_contiguous,
      input_bias_contiguous,
      hidden_bias_contiguous,
      cx_contiguous,
      hy,
      cy,
      workspace);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl__thnn_fused_lstm_cell_backward_impl(
    const c10::optional<at::Tensor>& grad_hy_opt,
    const c10::optional<at::Tensor>& grad_cy_opt,
    const Tensor& cx,
    const Tensor& cy,
    const Tensor& workspace,
    bool has_bias) {
  c10::MaybeOwned<Tensor> grad_hy_maybe_owned =
      at::borrow_from_optional_tensor(grad_hy_opt);
  const Tensor& grad_hy = *grad_hy_maybe_owned;
  const Tensor& grad_cy =
      c10::value_or_else(grad_cy_opt, [] { return Tensor(); });
  if (!grad_hy.defined() && !grad_cy.defined()) {
    return std::tuple<Tensor, Tensor, Tensor>();
  }
  checkLSTMBackwardSizesAndDtypes(
      {grad_hy, "grad_hy", 1},
      {grad_cy, "grad_cy", 2},
      {cx, "cx", 3},
      {cy, "cy", 4});
  const int gate_num = 4;
  const auto batch = cx.size(0);
  const auto hidden_size = cx.size(1);
  auto grad_hy_contiguous =
      grad_hy.defined() ? cnnl_contiguous(grad_hy) : grad_hy;
  auto grad_cy_contiguous =
      grad_cy.defined() ? cnnl_contiguous(grad_cy) : grad_cy;
  auto cx_contiguous = cnnl_contiguous(cx);
  auto cy_contiguous = cnnl_contiguous(cy);
  std::vector<int64_t> cnnl_gates_shape = {batch, gate_num, hidden_size};
  auto grad_gates = at::empty(
      cnnl_gates_shape,
      cx_contiguous.options(),
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_cx = at::empty_like(cx_contiguous, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  lstm_cell_backward_internal(
      grad_hy_contiguous,
      grad_cy_contiguous,
      cx_contiguous,
      cy_contiguous,
      workspace,
      grad_gates,
      grad_cx);
  // from [N, gate_num, hidden_size] to [N, gate_num*hidden_size]
  auto grad_gates_reshape =
      cnnl_contiguous(grad_gates.reshape({batch, gate_num * hidden_size}));

  auto grad_bias =
      has_bias ? grad_gates_reshape.sum(0, /*keepdim=*/false) : at::Tensor{};
  return std::make_tuple(grad_gates_reshape, grad_cx, grad_bias);
}

////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// GRU CELL
///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
static constexpr int64_t GRU_WORKSPACE_MULTIPLIER = 5;
std::tuple<Tensor, Tensor> cnnl__thnn_fused_gru_cell(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx,
    const c10::optional<Tensor>& input_bias_opt,
    const c10::optional<Tensor>& hidden_bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> input_bias_maybe_owned =
      at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias =
      c10::value_or_else(hidden_bias_opt, [] { return Tensor(); });
  const int gate_num = 3;
  const auto batch = input_gates.size(0);
  const auto hidden_size = input_gates.size(1) / gate_num;

  checkSizesAndDtypes(
      "cnnl__thnn_fused_gru_cell",
      {input_gates, "input_gates", 1},
      {hidden_gates, "hidden_gates", 2},
      {input_bias, "input_bias", 3},
      {hidden_bias, "hidden_bias", 4},
      /*factor=*/gate_num,
      {hx, "prev_hidden", 5});

  // reshape Tensor size from [N, gate_num*hidden_size] to [N, gate_num,
  // hidden_size]
  std::vector<int64_t> cnnl_gates_shape = {batch, gate_num, hidden_size};
  // reshape bias from [gate_num*hidden_size] to [gate_num, hidden_size]
  std::vector<int64_t> bias_shape = {gate_num, hidden_size};

  auto input_gates_contiguous =
      cnnl_contiguous(input_gates.reshape(cnnl_gates_shape));
  auto hidden_gates_contiguous =
      cnnl_contiguous(hidden_gates.reshape(cnnl_gates_shape));

  auto hx_contiguous = cnnl_contiguous(hx);
  auto input_bias_contiguous = input_bias.defined()
      ? cnnl_contiguous(input_bias).reshape(bias_shape)
      : input_bias;
  auto hidden_bias_contiguous = hidden_bias.defined()
      ? cnnl_contiguous(hidden_bias).reshape(bias_shape)
      : hidden_bias;
  auto hy = at::empty_like(hx_contiguous, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::Tensor workspace;
  workspace = at::empty(
      {hx.size(0), GRU_WORKSPACE_MULTIPLIER, hx.size(1)}, hx.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input_gates.scalar_type(), "cnnl__thnn_fused_gru_cell", [&] {
        gru_cell_forward_internal(
            input_gates_contiguous,
            hidden_gates_contiguous,
            input_bias_contiguous,
            hidden_bias_contiguous,
            hx_contiguous,
            hy,
            workspace);
      });
  return std::make_tuple(hy, workspace);
}

void checkGRUBackwardSizesAndDtypes(
    const at::TensorArg& grad_hy,
    const at::TensorArg& workspace) {
  at::CheckedFrom c = "fused_gru_cell_backward";
  at::checkDim(c, grad_hy, 2);
  at::checkSize(
      c,
      workspace,
      {grad_hy->size(0), GRU_WORKSPACE_MULTIPLIER, grad_hy->size(1)});
  at::checkAllSameType(c, {grad_hy, workspace});
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
cnnl__thnn_fused_gru_cell_backward(
    const Tensor& grad_hy,
    const Tensor& workspace,
    bool has_bias) {
  checkGRUBackwardSizesAndDtypes(
      {grad_hy, "grad_hy", 1}, {workspace, "workspace", 2});
  const int gate_num = 3;
  int64_t hidden_size = workspace.size(2);
  int64_t batch_size = workspace.size(0);
  auto grad_hy_contiguous = cnnl_contiguous(grad_hy);
  auto workspace_contiguous = cnnl_contiguous(workspace);
  auto grad_input_gates = at::empty(
      {batch_size, gate_num, hidden_size}, workspace_contiguous.options());
  auto grad_hidden_gates = at::empty(
      {batch_size, gate_num, hidden_size}, workspace_contiguous.options());
  auto grad_hx =
      at::empty_like(grad_hy_contiguous, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_hy.scalar_type(), "cnnl__thnn_fused_gru_cell_backward", [&] {
        gru_cell_backward_internal(
            grad_hy_contiguous,
            workspace_contiguous,
            grad_input_gates,
            grad_hidden_gates,
            grad_hx);
      });

  // from [N, gate_num, hidden_size] to [N, gate_num*hidden_size]
  auto grad_input_gates_reshape = cnnl_contiguous(
      grad_input_gates.reshape({batch_size, gate_num * hidden_size}));
  auto grad_hidden_gates_reshape = cnnl_contiguous(
      grad_hidden_gates.reshape({batch_size, gate_num * hidden_size}));
  at::Tensor grad_input_bias, grad_hidden_bias;
  if (has_bias) {
    grad_input_bias = grad_input_gates_reshape.sum(0, /*keepdim=*/false);
    grad_hidden_bias = grad_hidden_gates_reshape.sum(0, /*keepdim=*/false);
  }
  return std::make_tuple(
      grad_input_gates_reshape,
      grad_hidden_gates_reshape,
      grad_hx,
      grad_input_bias,
      grad_hidden_bias);
}

////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// LSTM
////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

using params_type = std::vector<std::vector<at::Tensor>>;
using hidden_type = std::vector<std::tuple<at::Tensor, at::Tensor>>;
// Parses a flat list of parameter tensors into a list of vector
static params_type gather_params_as_vector(
    const TensorList& params,
    const bool has_biases,
    const bool bidirectional,
    const bool has_projections) {
  params_type result;
  if (has_biases) {
    size_t each_size = has_projections ? 5 : 4;
    each_size = bidirectional ? (each_size * 2) : each_size;
    TORCH_CHECK(
        params.size() % each_size == 0,
        "got an incorrect number of RNN parameters");
    const auto num_layers = params.size() / each_size;
    for (size_t i = 0; i < num_layers; ++i) {
      std::vector<at::Tensor> temp;
      temp.reserve(each_size);
      for (size_t j = 0; j < each_size; ++j) {
        temp.emplace_back(params[i * each_size + j]);
      }
      result.emplace_back(temp);
    }
  } else {
    size_t each_size = has_projections ? 3 : 2;
    each_size = bidirectional ? each_size * 2 : each_size;
    TORCH_CHECK(
        params.size() % each_size == 0,
        "got an incorrect number of RNN parameters");
    const auto num_layers = params.size() / each_size;
    for (size_t i = 0; i < num_layers; ++i) {
      std::vector<at::Tensor> temp;
      temp.reserve(each_size);
      for (size_t j = 0; j < each_size; ++j) {
        temp.emplace_back(params[i * each_size + j]);
      }
      result.emplace_back(temp);
    }
  }
  return result;
}

static hidden_type gather_hidden_as_vector(
    const std::tuple<Tensor, Tensor>& hidden,
    const int64_t num_layers) {
  hidden_type result;
  Tensor hx, cx;
  std::tie(hx, cx) = hidden;
  TORCH_CHECK(hx.size(0) == cx.size(0), "cx and hx need to equal.");
  TORCH_CHECK(
      hx.size(0) % num_layers == 0,
      "first dimension size of hidden hx and num_layers.");
  const auto times = hx.size(0) / num_layers;
  result.reserve(num_layers);
  for (size_t i = 0; i < num_layers; ++i) {
    at::Tensor layer_hx = at::slice(hx, 0, i * times, (i + 1) * times, 1);
    at::Tensor layer_cx = at::slice(cx, 0, i * times, (i + 1) * times, 1);
    result.emplace_back(
        std::make_tuple(std::move(layer_hx), std::move(layer_cx)));
  }
  return result;
}

// Follow the native cudnn_is_acceptable funtion logic.
bool rnn_fusion_is_acceptable(const at::Tensor& self) {
  if (!torch_mlu::Global::instance().allowOpFusion())
    return false;
  auto st = self.scalar_type();
  if (!(st == at::kDouble || st == at::kFloat || st == at::kHalf))
    return false;
  if (self.numel() == 0)
    return false;
  return true;
}

// call cnnl rnn, only support lstm.
// 1) mode only support LSTM;
// 2) Not support dropout;
// 3) weight_buf need support flatten_weight, now not support;
// 4) CNNL padded LSTM only support TNC.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__cudnn_rnn(
    const at::Tensor& input_r,
    TensorList weight,
    int64_t weight_stride0,
    const c10::optional<at::Tensor>& weight_buf_opt,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& cx_opt,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    const c10::optional<at::Tensor>& dropout_state_opt) {
  const at::Tensor& weight_buf =
      *at::borrow_from_optional_tensor(weight_buf_opt);
  const at::Tensor& cx = c10::value_or_else(cx_opt, [] { return Tensor(); });
  const at::Tensor& dropout_state =
      c10::value_or_else(dropout_state_opt, [] { return Tensor(); });

  auto input = input_r;
  check_attributes(input, weight, {hx, cx}, true);

  auto is_packed_input = batch_sizes.size() != 0;
  // Packed input just two dims, size like: (real tokens num, feature dim)
  if (batch_first && !is_packed_input) {
    input = input.transpose(0, 1);
  }

  TORCH_CHECK(hx.is_contiguous(), "cnnl rnn: hx is not contiguous");
  TORCH_CHECK(
      !cx.defined() || cx.is_contiguous(), "cnnl rnn: cx is not contiguous");

  auto x = input.contiguous();

  if (proj_size != 0) {
    --weight_stride0;
  }
  bool has_biases = weight_stride0 == 4 ? true : false;

  auto output = cnnl_rnn_training_internal(
      x,
      hx,
      cx,
      weight,
      has_biases,
      (cnnlRNNMode_t)mode,
      hidden_size,
      proj_size,
      num_layers,
      bidirectional,
      train,
      batch_sizes);

  if (batch_first && !is_packed_input) {
    std::get<0>(output).transpose_(0, 1);
  }

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>
cnnl__cudnn_rnn_backward(
    const at::Tensor& input,
    TensorList weight,
    int64_t weight_stride0,
    const at::Tensor& weight_buf,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& cx_opt,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& grad_output_opt,
    const c10::optional<at::Tensor>& grad_hy_opt,
    const c10::optional<at::Tensor>& grad_cy_opt,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    const c10::optional<at::Tensor>& dropout_state_opt,
    const at::Tensor& reserve,
    std::array<bool, 4ul> output_mask) {
  const at::Tensor& cx = *at::borrow_from_optional_tensor(cx_opt);
  const at::Tensor& grad_output =
      c10::value_or_else(grad_output_opt, [] { return Tensor(); });
  const at::Tensor& grad_hy =
      c10::value_or_else(grad_hy_opt, [] { return Tensor(); });
  const at::Tensor& grad_cy =
      c10::value_or_else(grad_cy_opt, [] { return Tensor(); });
  const at::Tensor& dropout_state =
      c10::value_or_else(dropout_state_opt, [] { return Tensor(); });

  if (!grad_output.defined() && !grad_hy.defined() && !grad_cy.defined()) {
    return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>(
        Tensor(), Tensor(), Tensor(), std::vector<Tensor>(weight.size()));
  }

  auto grad_output_r = grad_output.defined()
      ? grad_output
      : at::zeros_like(output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_hy_r = grad_hy.defined()
      ? grad_hy
      : at::zeros_like(hx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_cy_r = cx.defined()
      ? (grad_cy.defined()
             ? grad_cy
             : at::zeros_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
      : grad_cy;

  const bool is_packed_input = batch_sizes.size() != 0;
  int* batch_sizes_int_ptr = nullptr;
  std::vector<int> batch_sizes_int;
  // convert cpu tensor from int64 to int.
  if (is_packed_input) {
    batch_sizes_int.reserve(batch_sizes.size());
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
      // batch_sizes is store batch size num of packed input,
      // so no need to check cast overflow.
      batch_sizes_int.push_back(
          c10::checked_convert<int32_t, int64_t>(batch_sizes[i], "int32_t"));
    }
    batch_sizes_int_ptr = batch_sizes_int.data();
  }

  // dev seq arr space and copy seq arr from cpu to mlu.
  at::Tensor dev_batch_sizes;
  if (is_packed_input) {
    const size_t copy_size = batch_sizes.size() * sizeof(int);
    dev_batch_sizes = at::empty(
        {static_cast<long>(batch_sizes.size())},
        input.options().dtype(at::kInt));
    int* dev_seq_lengths_ptr = dev_batch_sizes.data_ptr<int>();
    auto stream = getCurrentMLUStream();
    CNRT_CHECK(cnrtMemcpyAsync_V2(
        (void*)dev_seq_lengths_ptr,
        (void*)batch_sizes_int_ptr,
        copy_size,
        stream.stream(),
        CNRT_MEM_TRANS_DIR_HOST2DEV));
    stream.synchronize();
  }

  at::Tensor input_trans = input;
  at::Tensor grad_output_trans = grad_output_r;
  at::Tensor output_trans = output;

  if (batch_first && !is_packed_input) {
    input_trans = input.transpose(0, 1);
    grad_output_trans = grad_output_r.transpose(0, 1);
    output_trans = output.transpose(0, 1);
  }
  at::Tensor input_contiguous = cnnl_contiguous(input_trans);
  at::Tensor grad_contiguous = cnnl_contiguous(grad_output_trans);
  at::Tensor output_contiguous = cnnl_contiguous(output_trans);

  at::Tensor dx, dhx, dcx;
  std::tie(dx, dhx, dcx) = cnnl_rnn_backward_input_internal(
      input_contiguous,
      weight_buf,
      weight_stride0,
      hx,
      cx,
      output_contiguous,
      grad_contiguous,
      grad_hy_r,
      grad_cy_r,
      mode,
      hidden_size,
      proj_size,
      num_layers,
      dropout,
      train,
      bidirectional,
      batch_sizes_int_ptr,
      dev_batch_sizes,
      dropout_state,
      reserve,
      {output_mask[0], output_mask[1], output_mask[2]});

  std::vector<Tensor> dw;
  if (output_mask[3]) {
    dw = cnnl_rnn_backward_weight_internal(
        input_contiguous,
        weight,
        weight_stride0,
        weight_buf,
        hx,
        cx,
        output_contiguous,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_sizes_int_ptr,
        dev_batch_sizes,
        dropout_state,
        reserve);
  }

  if (batch_first && !is_packed_input) {
    dx = dx.transpose_(0, 1);
  }
  return std::
      tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>{
          dx, dhx, dcx, dw};
}

// rnn impl
std::pair<Tensor, std::tuple<Tensor, Tensor>> cnnl_rnn_impl(
    const Tensor& input,
    const Tensor& _batch_sizes,
    const std::tuple<Tensor, Tensor>& hidden,
    TensorList params,
    bool has_biases,
    cnnlRNNMode_t mode,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  Tensor hx, cx;
  std::tie(hx, cx) = hidden;
  int64_t hidden_size = hx.size(2);
  int64_t proj_size = 0;
  // For LSTM models with projections hidden size could be different
  if (cx.defined() && cx.size(2) != hx.size(2)) {
    hidden_size = cx.size(2);
    proj_size = hx.size(2);
  }

  torch_mlu::mlu::OptionalMLUGuard guard(input.get_device());
  bool is_packed_input = _batch_sizes.defined();
  TORCH_CHECK(
      input.dim() == (is_packed_input ? 2 : 3),
      "RNN packed input dim need be equal to 2, padded input dim equal to 3.");

  at::IntArrayRef batch_sizes = {};
  if (is_packed_input) {
    TORCH_CHECK(_batch_sizes.dim() == 1, "batch_sizes tensor should be 1D");
    batch_sizes = {
        _batch_sizes.data_ptr<int64_t>(),
        static_cast<size_t>(_batch_sizes.size(0))};
  }

  int64_t num_params = has_biases ? 4 : 2;
  if (proj_size != 0) {
    ++num_params;
  }

  // prepare params and hidden for each layer.
  auto params_vec =
      gather_params_as_vector(params, has_biases, bidirectional, proj_size > 0);
  auto hidden_vec = gather_hidden_as_vector(hidden, num_layers);

  // running cnnl kernel for each layer.
  // cnnl kernel support num_layer == 1 and bidirectional is true or false.
  at::Tensor layer_input = input;
  std::vector<at::Tensor> hy, cy;

  // TODO(PYTORCH-9424): May be optimized by use LSTMCell.
  // cnnl kernel not support multilayer lstm, torch_mlu loop call cnnl kernel.
  for (int64_t i = 0; i < num_layers; ++i) {
    at::Tensor t_p;
    at::Tensor weight;
    auto output = at::_cudnn_rnn(
        layer_input,
        params_vec[i],
        num_params,
        weight,
        std::get<0>(hidden_vec[i]),
        std::get<1>(hidden_vec[i]),
        static_cast<int>(mode),
        hidden_size,
        /*proj_size*/ proj_size,
        /*num_layers*/ 1,
        batch_first,
        dropout_p,
        train,
        bidirectional,
        batch_sizes,
        t_p);
    layer_input = std::get<0>(output);
    hy.emplace_back(std::move(std::get<1>(output)));
    cy.emplace_back(std::move(std::get<2>(output)));
    if (dropout_p != 0 && train && i < num_layers - 1) {
      layer_input = at::dropout(layer_input, dropout_p, train);
    }
  }
  return {layer_input, std::make_tuple(at::cat(hy, 0), at::cat(cy, 0))};
}

// packed rnn dispatch
void lstm_packed_cnnl(
    Tensor& output,
    Tensor& hy,
    Tensor& cy,
    const Tensor& data,
    const Tensor& batch_sizes,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  auto result = cnnl_rnn_impl(
      data,
      batch_sizes,
      std::make_tuple(hx[0], hx[1]),
      params,
      has_biases,
      CNNL_LSTM,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      false);
  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

// pad rnn dispatch
void lstm_cnnl(
    at::Tensor& output,
    at::Tensor& hy,
    at::Tensor& cy,
    const at::Tensor& input,
    TensorList hx,
    TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  at::Tensor batch_sizes;
  auto result = cnnl_rnn_impl(
      input,
      batch_sizes,
      std::make_tuple(hx[0], hx[1]),
      params,
      has_biases,
      CNNL_LSTM,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);

  output = result.first;
  hy = std::get<0>(result.second);
  cy = std::get<1>(result.second);
}

std::tuple<Tensor, Tensor, Tensor> cnnl_lstm(
    const Tensor& _input,
    TensorList hx,
    TensorList _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
  if (rnn_fusion_is_acceptable(_input)) {
    Tensor output, hy, cy;
    lstm_cnnl(
        output,
        hy,
        cy,
        _input,
        hx,
        _params,
        has_biases,
        num_layers,
        dropout_p,
        train,
        bidirectional,
        batch_first);
    return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
  }
  return at::native::lstm(
      _input,
      hx,
      _params,
      has_biases,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);
}

std::tuple<Tensor, Tensor, Tensor> cnnl_lstm_input_autograd(
    const Tensor& _input,
    TensorList hx,
    TensorList _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional,
    bool batch_first) {
  return cnnl_lstm(
      _input,
      hx,
      _params,
      has_biases,
      num_layers,
      dropout_p,
      train,
      bidirectional,
      batch_first);
}

std::tuple<Tensor, Tensor, Tensor> cnnl_lstm(
    const Tensor& data,
    const Tensor& batch_sizes,
    TensorList hx,
    TensorList _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
  if (rnn_fusion_is_acceptable(data)) {
    Tensor output, hy, cy;
    lstm_packed_cnnl(
        output,
        hy,
        cy,
        data,
        batch_sizes,
        hx,
        _params,
        has_biases,
        num_layers,
        dropout_p,
        train,
        bidirectional);
    return std::make_tuple(std::move(output), std::move(hy), std::move(cy));
  }
  return at::native::lstm(
      data,
      batch_sizes,
      hx,
      _params,
      has_biases,
      num_layers,
      dropout_p,
      train,
      bidirectional);
}

std::tuple<Tensor, Tensor, Tensor> cnnl_lstm_data_autograd(
    const Tensor& data,
    const Tensor& batch_sizes,
    TensorList hx,
    TensorList _params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool train,
    bool bidirectional) {
  return cnnl_lstm(
      data,
      batch_sizes,
      hx,
      _params,
      has_biases,
      num_layers,
      dropout_p,
      train,
      bidirectional);
}

} // namespace ops
} // namespace torch_mlu
