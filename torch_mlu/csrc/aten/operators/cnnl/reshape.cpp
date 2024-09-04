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

#include <c10/util/Optional.h>

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ComplexHelper.h>
#include "ATen/InferSize.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

inline at::Tensor cnnl_view_impl(const at::Tensor& self, at::IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);
  if ((!stride.has_value()) && (self.dim() < 6) && (self.dim() > 3) &&
      (self.is_contiguous(get_channels_last_memory_format(self.dim())))) {
    // Add device guard for view op, view op logic is different with pytorch.
    // CATCH view op call cnnl permute op when input is cl contiguous.
    const torch_mlu::mlu::MLUGuard device_guard(self.device());
    auto self_channels_first =
        cnnl_contiguous(self, c10::MemoryFormat::Contiguous);
    inferred_size = at::infer_size(size, self_channels_first.numel());
    stride = at::detail::computeStride(
        self_channels_first.sizes(),
        self_channels_first.strides(),
        inferred_size);
    return at::native::_reshape_alias(
        self_channels_first, inferred_size, *stride);
  }
  TORCH_CHECK(
      stride.has_value(),
      "view size is "
      "not compatible with input tensor's size and stride (at least one dimension"
      " spans across two contiguous subspaces). Use .reshape(...) instead.");
  auto output = at::native::_reshape_alias(self, inferred_size, *stride);
  return output;
}

at::Tensor cnnl_view_as_complex(const at::Tensor& self) {
  return at::native::view_as_complex(self);
}

at::Tensor cnnl_view_as_real(const Tensor& self) {
  return at::native::view_as_real(self);
}

at::Tensor cnnl__reshape_alias(
    const at::Tensor& self,
    at::IntArrayRef sizes,
    at::IntArrayRef strides) {
  return at::native::_reshape_alias(self, sizes, strides);
}

at::Tensor cnnl_view(const at::Tensor& self, at::IntArrayRef size) {
  return cnnl_view_impl(self, size);
}

at::Tensor cnnl__unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  return cnnl_view_impl(self, size);
}

} // namespace ops
} // namespace torch_mlu
