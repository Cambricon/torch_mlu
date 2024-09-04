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

#include <ATen/native/TensorAdvancedIndexing.h>
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"
#include "aten/operators/cnnl/scatter_utils.h"

namespace torch_mlu {
namespace ops {

using at::native::ReductionType;
using at::native::scatter_add_stub;
using at::native::scatter_fill_stub;
using at::native::scatter_reduce_stub;
using at::native::scatter_reduce_two_stub;
using at::native::scatter_scalar_reduce_stub;
using at::native::scatter_stub;

void scatter_mlu_kernel(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
  auto memory_format = at::MemoryFormat::Contiguous;
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto src_contiguous = cnnl_contiguous(src, memory_format);
  cnnl_scatter_internal(
      self_contiguous,
      self_contiguous,
      dim,
      index,
      src_contiguous,
      CNNL_SCATTER);

  if (!self.is_same(self_contiguous)) {
    self.copy_(self_contiguous);
  }
}

void scatter_fill_mlu_kernel(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value) {
  // use scatter to realize scatter_fill
  auto ndim = self.dim();
  std::vector<int64_t> shape(ndim, 1);
  auto src = at::full(shape, value, self.options().device(at::kPrivateUse1));
  scatter_mlu_kernel(self, dim, index, src);
}

void scatter_reduce_mlu_kernel(
    const at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    const ReductionType& reduce) {
  auto memory_format = at::MemoryFormat::Contiguous;
  auto self_contiguous =
      cast_long_to_int_if_needed(cnnl_contiguous(self, memory_format));
  auto src_contiguous =
      cast_long_to_int_if_needed(cnnl_contiguous(src, memory_format));
  switch (reduce) {
    case ReductionType::SUM:
      cnnl_scatter_internal(
          self_contiguous,
          self_contiguous,
          dim,
          index,
          src_contiguous,
          CNNL_SCATTER_ADD);
      break;
    case ReductionType::MAX:
      cnnl_scatter_internal(
          self_contiguous,
          self_contiguous,
          dim,
          index,
          src_contiguous,
          CNNL_SCATTER_MAX);
      break;
    case ReductionType::MIN:
      cnnl_scatter_internal(
          self_contiguous,
          self_contiguous,
          dim,
          index,
          src_contiguous,
          CNNL_SCATTER_MIN);
      break;
    case ReductionType::PROD:
      TORCH_CHECK(false, "MLU scatter reduce of prod is not supported");
    case ReductionType::MEAN:
      TORCH_CHECK(false, "MLU scatter reduce of mean is not supported");
    default:
      break;
  }

  if (!self.is_same(self_contiguous)) {
    self.copy_(self_contiguous);
  }
}

void scatter_add_mlu_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  // complex, int8, uint8, int16, bool are not supported
  AT_DISPATCH_MLU_FLOAT_HALF_INT_AND_BFLOAT16(
      self.scalar_type(), "MLU scatter_add", [&] {
        scatter_reduce_mlu_kernel(self, dim, index, src, ReductionType::SUM);
      });
}

void scatter_scalar_reduce_mlu_kernel(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value,
    const ReductionType& reduce) {
  auto ndim = self.dim();
  std::vector<int64_t> shape(ndim, 1);
  auto src = at::full(shape, value, self.options().device(at::kPrivateUse1));
  scatter_reduce_mlu_kernel(self, dim, index, src, reduce);
}

void scatter_reduce_two_mlu_kernel(
    const at::Tensor& self,
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    const ReductionType& reduce) {
  scatter_reduce_mlu_kernel(self, dim, index, src, reduce);
}

REGISTER_PRIVATEUSE1_DISPATCH(scatter_stub, &scatter_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(scatter_fill_stub, &scatter_fill_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(scatter_add_stub, &scatter_add_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(scatter_reduce_stub, &scatter_reduce_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    scatter_scalar_reduce_stub,
    &scatter_scalar_reduce_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    scatter_reduce_two_stub,
    &scatter_reduce_two_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
