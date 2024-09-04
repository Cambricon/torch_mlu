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
  auto dy_impl = getMluTensorImpl(dY);
  auto mean_impl = getMluTensorImpl(mean);
  auto rstd_impl = getMluTensorImpl(rstd);
  auto gamma_impl = getMluTensorImpl(gamma);
  auto dx_impl = getMluTensorImpl(dX);
  auto dgamma_impl = getMluTensorImpl(dgamma);
  auto dbeta_impl = getMluTensorImpl(dbeta);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor x_desc;
  CnnlTensorDescriptor dy_desc;
  CnnlTensorDescriptor mean_desc;
  CnnlTensorDescriptor rstd_desc;
  CnnlTensorDescriptor gamma_desc;
  CnnlTensorDescriptor dx_desc;
  CnnlTensorDescriptor dgamma_desc;
  CnnlTensorDescriptor dbeta_desc;

  x_desc.set(x_contiguous, CNNL_LAYOUT_NCHW);
  dy_desc.set(dY, CNNL_LAYOUT_NCHW);
  dx_desc.set(dX, CNNL_LAYOUT_NCHW);
  mean_desc.set(mean, CNNL_LAYOUT_ARRAY);
  rstd_desc.set(rstd, CNNL_LAYOUT_ARRAY);
  gamma_desc.set(gamma, CNNL_LAYOUT_ARRAY);
  dgamma_desc.set(dgamma, CNNL_LAYOUT_ARRAY);
  dbeta_desc.set(dbeta, CNNL_LAYOUT_ARRAY);

  auto x_ptr = x_impl->mlu_data_ptr();
  auto dy_ptr = dy_impl->mlu_data_ptr();
  auto mean_ptr = mean_impl->mlu_data_ptr();
  auto rstd_ptr = rstd_impl->mlu_data_ptr();
  auto gamma_ptr = gamma_impl->mlu_data_ptr();
  auto dx_ptr = dx_impl->mlu_data_ptr();
  auto dgamma_ptr = dgamma_impl->mlu_data_ptr();
  auto dbeta_ptr = dbeta_impl->mlu_data_ptr();

  // get workspace for GroupNormBackward
  int32_t NC = N * C;
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetGroupNormBackwardWorkspaceSizeV2(
      handle, x_desc.desc(), group, &workspace_size));

  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlGroupNormBackward(
      handle,
      x_desc.desc(),
      x_ptr,
      dy_desc.desc(),
      dy_ptr,
      gamma_desc.desc(),
      gamma_ptr,
      mean_desc.desc(),
      mean_ptr,
      rstd_desc.desc(),
      rstd_ptr,
      group,
      dx_desc.desc(),
      dx_ptr,
      dgamma_desc.desc(),
      dgamma_ptr,
      dbeta_desc.desc(),
      dbeta_ptr,
      ws_ptr.get(),
      workspace_size));
  return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace ops
} // namespace torch_mlu
