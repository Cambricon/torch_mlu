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

#include <ATen/MemoryOverlap.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {
at::Tensor& cnnl_masked_scatter_(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter: expected self and source to have same dtypes but got",
      self.scalar_type(),
      " and ",
      source.scalar_type());

  c10::MaybeOwned<Tensor> b_mask =
      expand_inplace(self, mask, "masked_scatter_");
  if (b_mask->dtype() == at::ScalarType::Byte) {
    TORCH_WARN(
        "masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated,"
        "please use a mask with dtype torch.bool instead.");
  }

  if (self.numel() == 0) {
    return self;
  }

  // Only Char, Bool and Byte masks are legal
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte ||
          mask.scalar_type() == at::ScalarType::Bool,
      "masked_scatter: expected BoolTensor or ByteTensor for mask");

  at::Tensor mask_tmp = *b_mask;
  auto memory_format = c10::MemoryFormat::Contiguous;
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto mask_contiguous = cnnl_contiguous(mask_tmp, memory_format);
  auto source_contiguous = cnnl_contiguous(source, memory_format);
  AT_DISPATCH_FLOATING_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Long,
      self_contiguous.scalar_type(),
      "masked_scatter_internal",
      [&] {
        cnnl_masked_scatter_internal(
            self_contiguous,
            self_contiguous,
            mask_contiguous,
            source_contiguous);
        TORCH_CNRT_CHECK(cnrtGetLastError());
      });
  if (is_copy_necessary(self, self_contiguous)) {
    self.copy_(self_contiguous);
  }
  return self;
}

} // namespace ops
} // namespace torch_mlu
