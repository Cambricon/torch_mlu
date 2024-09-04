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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"
#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_binary_cross_entropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  Tensor loss = at::empty({0}, self.options());
  return cnnl_binary_cross_entropy_out(self, target, weight, reduction, loss);
}

at::Tensor& cnnl_binary_cross_entropy_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    at::Tensor& out) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  at::TensorIteratorConfig config;
  config.add_output(out).add_input(self).add_input(target);
  if (weight.defined())
    config.add_input(weight);
  auto iter = config.build();
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "bce");
  auto output = iter.output(0);
  if (reduction != at::Reduction::None) {
    resize_impl_mlu_(getMluTensorImpl(output), {}, {});
  }
  cnnl_bce_internal(
      output,
      iter.input(0),
      iter.input(1),
      weight.defined() ? iter.input(2) : weight,
      reduction);
  iter_bridge.cast_outputs(iter);
  return out;
}

at::Tensor cnnl_binary_cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  Tensor grad_input = at::empty({0}, self.options());
  return cnnl_binary_cross_entropy_backward_out(
      grad_output, self, target, weight, reduction, grad_input);
}

at::Tensor& cnnl_binary_cross_entropy_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    at::Tensor& grad_input) {
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  at::TensorIteratorConfig config;
  config.add_output(grad_input)
      .add_input(grad_output)
      .add_input(self)
      .add_input(target);
  if (weight.defined())
    config.add_input(weight);
  auto iter = config.build();
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "bce");
  auto output = iter.output(0);
  cnnl_bce_bp_internal(
      output,
      iter.input(0),
      iter.input(1),
      iter.input(2),
      weight.defined() ? iter.input(3) : weight,
      reduction);
  iter_bridge.cast_outputs(iter);
  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
