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

#include "ATen/ops/empty.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"
#include "aten/operators/cnnl/resize.h"
#include "c10/core/ScalarType.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_nonzero_out(const at::Tensor& self, at::Tensor& out) {
  TORCH_CHECK(
      self.numel() < std::numeric_limits<int>::max(),
      "nonzero is not supported for tensors with more than INT_MAX elements, file a support request");
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Long,
      "the datatype of out in cnnl_nonzero_out "
      "must be Long, but got ",
      out.scalar_type())
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= CNNL_DIM_MAX,
      "nonzero is not supported for tensor with more than ",
      CNNL_DIM_MAX,
      " dimensions");
  if (self.numel() == 0) {
    resize_impl_mlu_(getMluTensorImpl(out), {0, self.dim()}, c10::nullopt);
    return out;
  }
  if (!out.defined()) {
    auto temp = at::empty(0, self.options().dtype(at::ScalarType::Long));
    out.copy_(temp);
  }
  AT_DISPATCH_ALL_TYPES_AND3(
      at::kBool,
      at::kHalf,
      at::kBFloat16,
      self.scalar_type(),
      "cnnl_nonzero",
      [&] {
        auto self_contiguous =
            cast_long_to_int_if_needed(cnnl_contiguous(self));
        auto out_contiguous = cnnl_contiguous(out);
        cnnl_nonzero_internal(out_contiguous, self_contiguous);
        if (is_copy_necessary(out, out_contiguous)) {
          out.copy_(out_contiguous);
        }
      });
  return out;
}

at::Tensor cnnl_nonzero(const at::Tensor& self) {
  at::Tensor out =
      at::empty({0, self.dim()}, self.options().dtype(at::ScalarType::Long));
  cnnl_nonzero_out(self, out);
  return out;
}

} // namespace ops
} // namespace torch_mlu
