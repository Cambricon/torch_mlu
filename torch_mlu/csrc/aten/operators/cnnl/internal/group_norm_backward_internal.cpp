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
#include <cmath>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_group_norm_backward_internal(
    const at::Tensor& x_contiguous,
    const at::Tensor& dY,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    at::Tensor& dX,
    at::Tensor& dgamma,
    at::Tensor& dbeta) {
  if (dX.numel() == 0) {
    if (dgamma.defined()) {
      if (N == 0) {
        dgamma.fill_(0);
      } else {
        dgamma.fill_(NAN);
      }
    }
    if (dbeta.defined()) {
      dbeta.fill_(0);
    }
    return std::make_tuple(dX, dgamma, dbeta);
  }

  auto x_impl = getMluTensorImpl(x_contiguous);
  auto x_desc = getTensorDesc(x_impl, CNNL_LAYOUT_NCHW);
  auto x_ptr = mlu_data_ptr(x_impl);

  auto dy_impl = getMluTensorImpl(dY);
  auto dy_desc = getTensorDesc(dy_impl, CNNL_LAYOUT_NCHW);
  auto dy_ptr = mlu_data_ptr(dy_impl);

  auto dx_impl = getMluTensorImpl(dX);
  auto dx_desc = getTensorDesc(dx_impl, CNNL_LAYOUT_NCHW);
  auto dx_ptr = mlu_data_ptr(dx_impl);

  auto mean_impl = getMluTensorImpl(mean);
  auto mean_desc = getTensorDesc(mean_impl, CNNL_LAYOUT_ARRAY);
  auto mean_ptr = mlu_data_ptr(mean_impl);

  auto rstd_impl = getMluTensorImpl(rstd);
  auto rstd_desc = getTensorDesc(rstd_impl, CNNL_LAYOUT_ARRAY);
  auto rstd_ptr = mlu_data_ptr(rstd_impl);

  auto gamma_impl = getMluTensorImpl(gamma);
  auto gamma_desc = getTensorDesc(gamma_impl, CNNL_LAYOUT_ARRAY);
  auto gamma_ptr = mlu_data_ptr(gamma_impl);

  auto dgamma_impl = getMluTensorImpl(dgamma);
  auto dgamma_desc = getTensorDesc(dgamma_impl, CNNL_LAYOUT_ARRAY);
  auto dgamma_ptr = mlu_data_ptr(dgamma_impl);

  auto dbeta_impl = getMluTensorImpl(dbeta);
  auto dbeta_desc = getTensorDesc(dbeta_impl, CNNL_LAYOUT_ARRAY);
  auto dbeta_ptr = mlu_data_ptr(dbeta_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // get workspace for GroupNormBackward
  const int32_t NC = N * C;
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetGroupNormBackwardWorkspaceSize_v2(
      handle, x_desc.get(), group, &workspace_size));

  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlGroupNormBackward(
      handle,
      x_desc.get(),
      x_ptr,
      dy_desc.get(),
      dy_ptr,
      gamma_desc.get(),
      gamma_ptr,
      mean_desc.get(),
      mean_ptr,
      rstd_desc.get(),
      rstd_ptr,
      group,
      dx_desc.get(),
      dx_ptr,
      dgamma_desc.get(),
      dgamma_ptr,
      dbeta_desc.get(),
      dbeta_ptr,
      ws_ptr.get(),
      workspace_size));
  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace ops
} // namespace torch_mlu
