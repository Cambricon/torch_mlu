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

std::tuple<at::Tensor, at::Tensor> cnnl__weight_norm_interface_backward(
    const at::Tensor& grad_w,
    const at::Tensor& saved_v,
    const at::Tensor& saved_g,
    const at::Tensor& saved_norms,
    int64_t dim) {
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");
  TORCH_CHECK(
      dim == 0 || dim == saved_v.dim() - 1,
      "fused kernels can only be applied for first or last dim")

  auto grad_v = at::empty_like(saved_v, at::MemoryFormat::Contiguous);
  auto grad_g = at::empty_like(saved_g, at::MemoryFormat::Contiguous);
  auto grad_w_contiguous =
      cnnl_contiguous(grad_w, at::MemoryFormat::Contiguous);
  auto saved_v_contiguous =
      cnnl_contiguous(saved_v, at::MemoryFormat::Contiguous);
  auto saved_g_contiguous =
      cnnl_contiguous(saved_g, at::MemoryFormat::Contiguous);
  auto saved_norms_contiguous =
      cnnl_contiguous(saved_norms, at::MemoryFormat::Contiguous);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      saved_v.scalar_type(),
      "cnnl__weight_norm_interface_backward",
      [&] {
        cnnl_weight_norm_backward_internal(
            grad_v,
            grad_g,
            grad_w_contiguous,
            saved_v_contiguous,
            saved_g_contiguous,
            saved_norms_contiguous,
            dim);
      });
  return std::tuple<Tensor, Tensor>{grad_v, grad_g};
}

} // namespace ops
} // namespace torch_mlu
