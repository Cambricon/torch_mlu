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

#include "ATen/MemoryOverlap.h"
#include "ATen/core/ivalue_inl.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/tensor_util.h"

namespace torch_mlu {
namespace ops {

// Type list supported by GPU.
// coda path: aten/src/ATen/native/cuda/Indexing.cu
// index type list:        at::ScalarType::Int, at::ScalarType::Long
// dim type list:          int64_t
// input/output type list: at::ScalarType::Byte, at::ScalarType::Char,
//                         at::ScalarType::Int, at::ScalarType::Long,
//                         at::ScalarType::Short, at::ScalarType::Double,
//                         at::ScalarType::Float, at::ScalarType::ComplexDouble,
//                         at::ScalarType::ComplexFloat,
//                         at::ScalarType::ComplexHalf, at::ScalarType::Half,
//                         at::ScalarType::Bool, at::ScalarType::BFloat16.

// Type list supported by CNNL index select.
// index type list:        at::ScalarType::Int, at::ScalarType::Long
// dim type list:          int32
// input/output type list: at::ScalarType::Byte, at::ScalarType::Char,
//                         at::ScalarType::Short, at::ScalarType::Int,
//                         at::ScalarType::Half, at::ScalarType::Bool,
//                         at::ScalarType::Float, at::ScalarType::Long
//                         at::ScalarType::Double
// Not real support double, but support double to float convert.

void index_select_out_mlu_impl(
    at::Tensor& out,
    const at::Tensor& self,
    long dim,
    const at::Tensor& index) {
  ptrdiff_t numIndices = index.numel();
  int selfDims = self.dim() == 0 ? 1 : self.dim();

  TORCH_CHECK(
      index.dim() <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(dim < selfDims, "Indexing dim is out of bounds");

  std::vector<int64_t> newSize = self.sizes().vec();
  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }

  at::native::resize_output(out, newSize);

  ptrdiff_t outTotalSize = out.numel();
  if (outTotalSize == 0) {
    return;
  }

  // Get contiguous tensor for cnnl kernel
  auto self_contiguous = cnnl_contiguous(self, c10::MemoryFormat::Contiguous);
  auto index_contiguous = cnnl_contiguous(index, c10::MemoryFormat::Contiguous);
  auto output_contiguous = cnnl_contiguous(out, c10::MemoryFormat::Contiguous);
  // Call cnnl internal kernel, and index only support int and long type.
  AT_DISPATCH_INDEX_TYPES(
      index_contiguous.scalar_type(), "index_select_out_mlu_impl", [&]() {
        cnnl_index_select_internal(
            output_contiguous, self_contiguous, dim, index_contiguous);
      });
  if (is_copy_necessary(output_contiguous, out)) {
    out.copy_(output_contiguous);
  }
}

at::Tensor cnnl_index_select(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  at::Tensor out = at::empty({0}, self.options());
  cnnl_index_select_out(self, dim, index, out);
  return out;
}

at::Tensor& cnnl_index_select_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Tensor& out) {
  static constexpr c10::string_view DIM_WARNING =
      "Tensor too large or too many (> 8) dimensions";
  TORCH_CHECK(
      torch_mlu::check_device({out, self, index}),
      "Input, output and indices must be on the current device");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= CNNL_MAX_DIM_SIZE, DIM_WARNING);
  TORCH_CHECK(index.dim() <= CNNL_MAX_DIM_SIZE, DIM_WARNING);
  TORCH_CHECK(!self.is_quantized(), "torch_mlu not support quantized tensor.");
  // Implicit check in TensorInfo constructor in gpu side.
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      "self and out scalar type is not same, self scalar type: ",
      self.scalar_type(),
      " out scalar type: ",
      out.scalar_type());
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      out.scalar_type(),
      "index_select_mlu",
      [&] { index_select_out_mlu_impl(out, self, dim, index); });

  return out;
}

} // namespace ops
} // namespace torch_mlu
