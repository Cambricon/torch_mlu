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

std::tuple<Tensor, Tensor, Tensor> cnnl_native_group_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<Tensor>& gamma_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
  const at::Tensor& gamma = *at::borrow_from_optional_tensor(gamma_opt);
  at::Tensor gamma_contiguous;
  if (!gamma.defined()) {
    gamma_contiguous = at::ones({C}, X.options());
  } else {
    gamma_contiguous = cnnl_contiguous(gamma);
  }
  auto dgamma = at::empty(gamma_contiguous.sizes(), X.options());
  auto dbeta = at::empty(gamma_contiguous.sizes(), X.options());

  auto memory_format = at::MemoryFormat::Contiguous;
  auto x_contiguous = cnnl_contiguous(X, memory_format);
  auto dy_contiguous = cnnl_contiguous(dY, memory_format);
  x_contiguous = cnnl_view(x_contiguous, {N, C, 1, HxW});
  dy_contiguous = cnnl_view(dy_contiguous, {N, C, 1, HxW});
  auto dX = at::empty(x_contiguous.sizes(), x_contiguous.options());
  cnnl_group_norm_backward_internal(
      x_contiguous,
      dy_contiguous,
      mean,
      rstd,
      gamma_contiguous,
      N,
      C,
      HxW,
      group,
      dX,
      dgamma,
      dbeta);
  dX = cnnl_view(dX, X.sizes());

  // TODO(liangyuefeng): cnnlGroupNormBackward is not supported grad_input_mask
  // yet.
  at::Tensor dX_t;
  at::Tensor dgamma_t;
  at::Tensor dbeta_t;
  if (grad_input_mask[0]) {
    dX_t = dX;
  }
  if (grad_input_mask[1]) {
    dgamma_t = dgamma;
  }
  if (grad_input_mask[2]) {
    dbeta_t = dbeta;
  }
  return std::make_tuple(dX_t, dgamma_t, dbeta_t);
}

} // namespace ops
} // namespace torch_mlu
