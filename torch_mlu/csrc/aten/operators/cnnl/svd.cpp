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
#include <ATen/native/BatchLinearAlgebra.h>

namespace torch_mlu {
namespace ops {

#define MAX_HEIGHT 150
#define MAX_WIDTH 150
/* torch.svd, implemented in terms of torch.linalg.svd. There are two main
   differences:
    1. the 2nd parameter is bool some=True, which if effectively the opposite
       of full_matrices=True
    2. svd returns V, while linalg.svd returns Vh = V^H;
       If compute_uv is False, torch.svd() returns zero-filled tensors for U and
   Vh, whereas torch.linalg.svd() returns empty tensors.
*/
using at::native::svd_stub;
void svd_mlu_kernel(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const c10::optional<c10::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info) {
  auto size = A.sizes();
  auto dim = A.dim();
  TORCH_MLU_CHECK(
      size[dim - 1] < MAX_HEIGHT && size[dim - 2] < MAX_WIDTH,
      "The height or width of cnnl svd's input tensor should be less than 150");
  // TODO(CNNLCORE-13740): cnnl not support compute_uv's value  is false
  TORCH_CHECK(
      compute_uv, "torch_mlu svd not support compute_uv's value is false.");
  TORCH_CHECK(
      !driver.has_value(), "torch_mlu svd not support driver has value");
  at::Tensor s_contiguous = S;
  at::Tensor vh_contiguous = cnnl_contiguous(Vh);
  at::Tensor u_contiguous = cnnl_contiguous(U);
  std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> outputs(
      u_contiguous, s_contiguous, vh_contiguous);
  auto self_contiguous = cnnl_contiguous(A);
  cnnl_svd_internal(outputs, self_contiguous, !full_matrices, compute_uv, info);
  if (is_copy_necessary(U, u_contiguous)) {
    U.copy_(u_contiguous);
  }
  if (is_copy_necessary(S, s_contiguous)) {
    Vh.copy_(vh_contiguous);
  }
}

REGISTER_PRIVATEUSE1_DISPATCH(svd_stub, &svd_mlu_kernel);
} // namespace ops
} // namespace torch_mlu
