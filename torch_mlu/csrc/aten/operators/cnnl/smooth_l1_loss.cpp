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

#include <ATen/TensorIterator.h>
#include <ATen/core/Reduction.h>
#include "aten/TensorIteratorBridge.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_smooth_l1_loss(
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  auto output = at::empty({0}, input.options());
  return cnnl_smooth_l1_loss_out(input, target, reduction, beta, output);
}

at::Tensor& cnnl_smooth_l1_loss_out(
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta,
    at::Tensor& result) {
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")

  auto iter = at::TensorIterator::borrowing_binary_op(result, input, target);
  TORCH_INTERNAL_ASSERT(
      reduction == at::Reduction::None || reduction == at::Reduction::Mean ||
      reduction == at::Reduction::Sum);

  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "smooth_l1_loss");
  if (reduction == at::Reduction::Mean || reduction == at::Reduction::Sum) {
    result.resize_({});
  }

  auto output = iter.output(0);
  if (reduction == at::Reduction::Mean || reduction == at::Reduction::Sum) {
    output.resize_({});
  }

  cnnl_smooth_l1_loss_forward_internal(
      output, iter.input(0), iter.input(1), reduction, beta);
  iter_bridge.cast_outputs(iter);
  return result;
}

// register the backward operator here to avoid native zeros_like operator calls
// and make the implementation more consistent with the forward operator.
at::Tensor cnnl_smooth_l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  auto grad_input = at::empty({0}, self.options());
  return cnnl_smooth_l1_loss_backward_out(
      grad_output, self, target, reduction, beta, grad_input);
}

at::Tensor& cnnl_smooth_l1_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta,
    at::Tensor& grad_input) {
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(input)
                  .add_input(target)
                  .add_input(grad_output)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "smooth_l1_loss_backward");
  auto output = iter.output(0);
  cnnl_smooth_l1_loss_backward_internal(
      output, iter.input(2), iter.input(0), iter.input(1), reduction, beta);

  iter_bridge.cast_outputs(iter);
  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
