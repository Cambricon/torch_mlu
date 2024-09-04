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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_cdist_forward_internal(
    at::Tensor& result,
    const at::Tensor& x1,
    const at::Tensor& x2,
    const double p) {
  // TODO(PYTORCH-8665): cnnl_cdist only supports p = 1.0 for now.
  TORCH_CHECK(p == 1.0, "cnnl_cdist only supports p = 1.0 for now");
  auto result_impl = getMluTensorImpl(result);
  auto result_desc = getTensorDesc(result_impl);
  auto result_ptr = mlu_data_ptr(result_impl);

  auto x1_impl = getMluTensorImpl(x1);
  auto x1_desc = getTensorDesc(x1_impl);
  auto x1_ptr = mlu_data_ptr(x1_impl);

  auto x2_impl = getMluTensorImpl(x2);
  auto x2_desc = getTensorDesc(x2_impl);
  auto x2_ptr = mlu_data_ptr(x2_impl);

  // get current handle
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlCdistForward(
      handle,
      x1_desc.get(),
      x1_ptr,
      x2_desc.get(),
      x2_ptr,
      p,
      result_desc.get(),
      result_ptr));
  return result;
}

at::Tensor cnnl_cdist_backward_internal(
    at::Tensor& grad_x1,
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& cdist,
    const double p,
    const at::Tensor& grad) {
  auto grad_x1_impl = getMluTensorImpl(grad_x1);
  auto grad_x1_desc = getTensorDesc(grad_x1_impl);
  auto grad_x1_ptr = mlu_data_ptr(grad_x1_impl);

  auto x1_impl = getMluTensorImpl(x1);
  auto x1_desc = getTensorDesc(x1_impl);
  auto x1_ptr = mlu_data_ptr(x1_impl);

  auto x2_impl = getMluTensorImpl(x2);
  auto x2_desc = getTensorDesc(x2_impl);
  auto x2_ptr = mlu_data_ptr(x2_impl);

  auto cdist_impl = getMluTensorImpl(cdist);
  auto cdist_desc = getTensorDesc(cdist_impl);
  auto cdist_ptr = mlu_data_ptr(cdist_impl);

  auto grad_impl = getMluTensorImpl(grad);
  auto grad_desc = getTensorDesc(grad_impl);
  auto grad_ptr = mlu_data_ptr(grad_impl);

  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlCdistBackward(
      handle,
      x1_desc.get(),
      x1_ptr,
      x2_desc.get(),
      x2_ptr,
      cdist_desc.get(),
      cdist_ptr,
      grad_desc.get(),
      grad_ptr,
      p,
      grad_x1_desc.get(),
      grad_x1_ptr));
  return grad_x1;
}
} // namespace ops
} // namespace torch_mlu
