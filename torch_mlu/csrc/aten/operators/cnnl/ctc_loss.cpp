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

#include <torch/autograd.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

class CTCLossFunction : public torch::autograd::Function<CTCLossFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& probs,
      const at::Tensor& targets,
      const c10::optional<at::Tensor>& input_lengths_opt,
      const c10::optional<at::Tensor>& target_lengths_opt,
      at::IntArrayRef il,
      at::IntArrayRef tl,
      int64_t blank,
      int64_t reduction,
      bool zero_infinity,
      int64_t normalization) {
    at::AutoDispatchBelowADInplaceOrView g;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_mlu::ctc_loss_forward", "")
                         .typed<decltype(cnnl_ctc_loss_forward)>();
    auto result = op.call(
        probs,
        targets,
        input_lengths_opt,
        target_lengths_opt,
        il,
        tl,
        blank,
        reduction,
        zero_infinity,
        normalization);

    auto result0 = std::get<0>(result);
    auto result1 = std::get<1>(result);
    ctx->save_for_backward({result1});

    return {result0, result1};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto raw_grad = saved[0];
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_mlu::ctc_loss_backward", "")
                         .typed<decltype(cnnl_ctc_loss_backward)>();
    auto result = op.call(grad_outputs[0], raw_grad);
    return {
        result,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable()};
  }
};

// torch_mlu::ctc_loss_forward
std::tuple<at::Tensor, at::Tensor> cnnl_ctc_loss_forward(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const c10::optional<at::Tensor>& input_lengths_opt,
    const c10::optional<at::Tensor>& target_lengths_opt,
    at::IntArrayRef il,
    at::IntArrayRef tl,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity,
    int64_t normalization) {
  at::Tensor input_lengths, target_lengths, ilc, tlc;
  int64_t mode = 0;
  // for ctc_loss.Tensor
  if (input_lengths_opt.has_value() && target_lengths_opt.has_value()) {
    input_lengths = *at::borrow_from_optional_tensor(input_lengths_opt);
    target_lengths = *at::borrow_from_optional_tensor(target_lengths_opt);
    // get scalar value of input_lengths and target_lengths
    ilc = input_lengths.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
    tlc = target_lengths.to(at::Device(at::kCPU)).to(at::kLong).contiguous();
  }

  // for ctc_loss.IntList
  if (!input_lengths_opt.has_value() && !target_lengths_opt.has_value()) {
    // TODO(PYTORCH-8642) 2 extra H2D copies are used compare with cuda
    // ctc_loss.
    input_lengths = at::tensor(il).to(at::Device(at::kPrivateUse1));
    target_lengths = at::tensor(tl).to(at::Device(at::kPrivateUse1));
    mode = 1;
  }

  TORCH_CHECK(
      (input_lengths.scalar_type() == at::ScalarType::Long ||
       input_lengths.scalar_type() == at::ScalarType::Int),
      "input_lengths must be long or int");
  TORCH_CHECK(
      (target_lengths.scalar_type() == at::ScalarType::Long ||
       target_lengths.scalar_type() == at::ScalarType::Int),
      "target_lengths must be long or int");
  auto probs_contiguous = probs.device().is_privateuseone()
      ? cnnl_contiguous(probs)
      : cnnl_contiguous(probs.to(at::Device(at::kPrivateUse1)));
  auto targets_contiguous = targets.device().is_privateuseone()
      ? cnnl_contiguous(targets)
      : cnnl_contiguous(targets.to(at::Device(at::kPrivateUse1)));
  auto input_lengths_contiguous = input_lengths.device().is_privateuseone()
      ? cnnl_contiguous(input_lengths)
      : cnnl_contiguous(input_lengths.to(at::Device(at::kPrivateUse1)));
  input_lengths_contiguous =
      cast_long_to_int_if_needed(input_lengths_contiguous);
  auto target_lengths_contiguous = target_lengths.device().is_privateuseone()
      ? cnnl_contiguous(target_lengths)
      : cnnl_contiguous(target_lengths.to(at::Device(at::kPrivateUse1)));
  target_lengths_contiguous =
      cast_long_to_int_if_needed(target_lengths_contiguous);
  return cnnl_ctc_loss_internal(
      probs_contiguous,
      targets_contiguous,
      input_lengths_contiguous,
      target_lengths_contiguous,
      mode == 0 ? at::IntArrayRef(ilc.data_ptr<int64_t>(), ilc.numel()) : il,
      mode == 0 ? at::IntArrayRef(tlc.data_ptr<int64_t>(), tlc.numel()) : tl,
      blank,
      reduction,
      zero_infinity,
      normalization);
}

std::tuple<at::Tensor, at::Tensor> cnnl_ctc_loss_forward_autograd(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const c10::optional<at::Tensor>& input_lengths_opt,
    const c10::optional<at::Tensor>& target_lengths_opt,
    at::IntArrayRef il,
    at::IntArrayRef tl,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity,
    int64_t normalization) {
  auto result = CTCLossFunction::apply(
      probs,
      targets,
      input_lengths_opt,
      target_lengths_opt,
      il,
      tl,
      blank,
      reduction,
      zero_infinity,
      normalization);
  return std::make_tuple(result[0], result[1]);
}

// torch_mlu::warp_ctc_loss
at::Tensor cnnl_warp_ctc_loss(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity,
    int64_t normalization) {
  TORCH_CHECK(reduction == 1, "warp_ctc_loss only support sum mode.");
  TORCH_CHECK(
      normalization == 0,
      "warp_ctc_loss's input doesn't go through log_softmax.");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_mlu::ctc_loss_forward", "")
                       .typed<decltype(cnnl_ctc_loss_forward)>();
  auto result = op.call(
      probs,
      targets,
      input_lengths,
      target_lengths,
      {},
      {},
      blank,
      2, // reduction=sum
      zero_infinity,
      normalization); // normalization=none
  // loss is 1-dim for warp_ctc_loss and 0-dim for nn.CTCLoss
  return std::get<0>(result).unsqueeze(0);
}

at::Tensor cnnl_warp_ctc_loss_autograd(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity,
    int64_t normalization) {
  return cnnl_warp_ctc_loss(
      probs,
      targets,
      input_lengths,
      target_lengths,
      blank,
      reduction,
      zero_infinity,
      normalization);
}

// aten::ctc_loss.Tensor
at::Tensor cnnl_ctc_loss(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_mlu::ctc_loss_forward", "")
                       .typed<decltype(cnnl_ctc_loss_forward)>();
  auto result = op.call(
      probs,
      targets,
      input_lengths,
      target_lengths,
      {},
      {},
      blank,
      reduction,
      zero_infinity,
      2); // normalization=log_softmax
  return std::get<0>(result);
}

at::Tensor cnnl_ctc_loss_Tensor_autograd(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  return cnnl_ctc_loss(
      probs,
      targets,
      input_lengths,
      target_lengths,
      blank,
      reduction,
      zero_infinity);
}

// aten::ctc_loss.IntList
at::Tensor cnnl_ctc_loss(
    const at::Tensor& probs,
    const at::Tensor& targets,
    at::IntArrayRef il,
    at::IntArrayRef tl,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torch_mlu::ctc_loss_forward", "")
                       .typed<decltype(cnnl_ctc_loss_forward)>();
  auto result = op.call(
      probs,
      targets,
      c10::nullopt,
      c10::nullopt,
      il,
      tl,
      blank,
      reduction,
      zero_infinity,
      2); // normalization=log_softmax
  return std::get<0>(result);
}

at::Tensor cnnl_ctc_loss_IntList_autograd(
    const at::Tensor& probs,
    const at::Tensor& targets,
    at::IntArrayRef il,
    at::IntArrayRef tl,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  return cnnl_ctc_loss(probs, targets, il, tl, blank, reduction, zero_infinity);
}

at::Tensor cnnl_ctc_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& raw_grad) {
  auto grad_out_contiguous = cnnl_contiguous(grad_out);
  auto raw_grad_contiguous = cnnl_contiguous(raw_grad);
  if (grad_out.sizes().empty()) {
    return raw_grad_contiguous * grad_out_contiguous;
  } else {
    return raw_grad_contiguous * grad_out_contiguous.unsqueeze(0).unsqueeze(2);
  }
}

} // namespace ops
} // namespace torch_mlu
