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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_layer_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& gamma_opt,
    const c10::optional<at::Tensor>& beta_opt,
    std::array<bool, 3> grad_input_mask) {
  const int normalized_ndim = normalized_shape.size();

  c10::MaybeOwned<at::Tensor> gamma_maybe_owned =
      at::borrow_from_optional_tensor(gamma_opt);
  const at::Tensor& gamma = *gamma_maybe_owned;
  c10::MaybeOwned<at::Tensor> beta_maybe_owned =
      at::borrow_from_optional_tensor(beta_opt);
  const at::Tensor& beta = *beta_maybe_owned;

  at::native::_check_layer_norm_inputs(X, normalized_shape, gamma, beta);

  at::Tensor dX;
  at::Tensor dgamma;
  at::Tensor dbeta;

  at::Tensor gamma_contiguous;
  // TODO(miaochen): remove at::ones when PYTORCH-9358 finish
  if (!gamma.defined()) {
    gamma_contiguous = at::ones(normalized_shape, X.options());
  } else {
    gamma_contiguous = cnnl_contiguous(gamma, at::MemoryFormat::Contiguous);
  }

  auto axis = X.dim() - normalized_ndim;
  auto X_contiguous = cnnl_contiguous(X, at::MemoryFormat::Contiguous);
  auto dY_contiguous = cnnl_contiguous(dY, at::MemoryFormat::Contiguous);

  auto dX_contiguous = at::empty(X_contiguous.sizes(), X.options());
  auto dgamma_contiguous = at::empty(gamma_contiguous.sizes(), X.options());
  auto dbeta_contiguous = at::empty(gamma_contiguous.sizes(), X.options());

  if (X_contiguous.numel() > 0) {
    cnnl_native_layer_norm_backward_internal(
        dY_contiguous,
        X_contiguous,
        axis,
        cnnl_contiguous(mean),
        cnnl_contiguous(rstd),
        gamma_contiguous,
        dX_contiguous,
        dgamma_contiguous,
        dbeta_contiguous);
  } else {
    if (dgamma_contiguous.numel() > 0) {
      dgamma_contiguous.zero_();
      dbeta_contiguous.zero_();
    }
  }

  if (grad_input_mask[0]) {
    dX = dX_contiguous;
  }
  if (grad_input_mask[1]) {
    dgamma = dgamma_contiguous;
  }
  if (grad_input_mask[2]) {
    dbeta = dbeta_contiguous;
  }

  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace ops
} // namespace torch_mlu
