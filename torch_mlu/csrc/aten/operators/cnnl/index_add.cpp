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
#include "ATen/MemoryOverlap.h"
#include "ATen/core/ivalue_inl.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {
void index_add_mlu_impl(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha,
    const at::Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }
  const at::Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const at::Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(
      result.dim() <= CNNL_MAX_DIM_SIZE,
      "tensor has too many (>",
      CNNL_MAX_DIM_SIZE,
      ") dims");
  TORCH_CHECK(
      source.dim() <= CNNL_MAX_DIM_SIZE,
      "tensor has too many (>",
      CNNL_MAX_DIM_SIZE,
      ") dims");
  TORCH_CHECK(
      index.dim() <= CNNL_MAX_DIM_SIZE,
      "tensor has too many (>",
      CNNL_MAX_DIM_SIZE,
      ") dims");

  auto memory_format = c10::MemoryFormat::Contiguous;
  auto self_contiguous = cnnl_contiguous(self_, memory_format);
  auto index_contiguous = cnnl_contiguous(index, memory_format);
  auto result_contiguous = (result.dim() == 0) ? result : self_contiguous;
  // TODO(PYTORCH-9295): cnnl_index_add does not support alpha, we use mul +
  // index_add instead.
  at::Tensor source_contiguous;
  if (alpha.to<float>() != 1.0) {
    source_contiguous =
        cnnl_contiguous(cnnl_mul(source_, alpha), memory_format);
  } else {
    source_contiguous = cnnl_contiguous(source_, memory_format);
  }

  cnnl_index_add_internal(
      result_contiguous,
      self_contiguous,
      dim,
      index_contiguous,
      source_contiguous);
  if (is_copy_necessary(result, result_contiguous)) {
    result.copy_(result_contiguous);
  }
}

TORCH_IMPL_FUNC(index_add_out_mlu)
(const at::Tensor& self,
 int64_t dim,
 const at::Tensor& index,
 const at::Tensor& source,
 const at::Scalar& alpha,
 const at::Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "index_add",
      [&] { index_add_mlu_impl(self, dim, index, source, alpha, result); });
}
} // namespace ops
} // namespace torch_mlu
