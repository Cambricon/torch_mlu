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
#include <torch/library.h>

namespace torch_mlu {
namespace ops {

// To be registered for the "_cudnn_rnn(...)" schema.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_cnnl_rnn_cast(
    const at::Tensor& input,
    at::TensorList weight,
    int64_t weight_stride0,
    const c10::optional<at::Tensor>& weight_buf_opt,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& cx,
    int64_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool batch_first,
    double dropout,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    const c10::optional<at::Tensor>& dropout_state) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
      c10::DispatchKey::AutocastPrivateUse1);

  for (const auto& t : weight) {
    TORCH_CHECK(
        weight[0].scalar_type() == t.scalar_type(),
        "Weight scalar types do not match.");
  }
  // weight_stride0 is the number of weight tensors per layer and direction, as
  // seen by model.parameters(). If bias is enabled, there are 4 such tensors
  // (ih and hh weights, ih and hh biases). If bias is not enabled, there are 2
  // (ih and hh weights). This organization holds for all rnn types (RNN, GRU,
  // and LSTM). If LSTM with projections is used, additional hr weight is added.
  if (proj_size > 0) {
    TORCH_INTERNAL_ASSERT(
        (weight_stride0 == 3) || (weight_stride0 == 5),
        "weight_stride0 must be 3 (if no bias) or 5 (if bias) for LSTM with projections.  Received ",
        weight_stride0);
  } else {
    TORCH_INTERNAL_ASSERT(
        (weight_stride0 == 2) || (weight_stride0 == 4),
        "weight_stride0 must be 2 (if no bias) or 4 (if bias).  Received ",
        weight_stride0);
  }

  //  Imitate aten/src/ATen/cudnn/AutocastRNN.cpp:_cudnn_rnn_cast_reflatten
  //  weight_buf_opt is not support in catch LSTM.
  bool needs_cast =
      at::autocast::is_eligible(weight[0], at::DeviceType::PrivateUse1) &&
      (weight[0].scalar_type() != at::kHalf);

  std::vector<at::Tensor> weight_casted;
  if (needs_cast) {
    weight_casted.reserve(weight.size());
    for (auto& each : weight) {
      weight_casted.push_back(each.to(at::kHalf));
    }
  }

  return at::_cudnn_rnn(
      at::autocast::cached_cast(at::kHalf, input, at::DeviceType::PrivateUse1),
      needs_cast ? at::TensorList(weight_casted) : weight,
      weight_stride0,
      weight_buf_opt,
      at::autocast::cached_cast(at::kHalf, hx, at::DeviceType::PrivateUse1),
      at::autocast::cached_cast(at::kHalf, cx, at::DeviceType::PrivateUse1),
      mode,
      hidden_size,
      proj_size,
      num_layers,
      batch_first,
      dropout,
      train,
      bidirectional,
      batch_sizes,
      dropout_state);
}

namespace {
TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  m.impl("_cudnn_rnn", TORCH_FN((&torch_mlu::ops::_cnnl_rnn_cast)));
}
} // anonymous namespace

} // namespace ops
} // namespace torch_mlu
