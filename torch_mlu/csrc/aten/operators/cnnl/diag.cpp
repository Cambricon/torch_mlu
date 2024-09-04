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
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/types.h"
#include <ATen/MemoryOverlap.h>

namespace torch_mlu {
namespace ops {

// Type list supported by native diag
// cuda path: aten/sec/ATen/native/TensorShape
// iter.common_dtype()
// at::ScalarType::Bool
// at::ScalarType::Byte
// at::ScalarType::Char
// at::ScalarType::Int
// at::ScalarType::Long
// at::ScalarType::Short
// at::ScalarType::Float
// at::ScalarType::Double
// at::ScalarType::ComplexFloat
// at::ScalarType::ComplexDouble
// at::ScalarType::BFloat16
// at::ScalarType::Half
// at::ScalarType::ComplexHalf
// at::ScalarType::kBool
// at::ScalarType::KBfloat16

template <typename scalar_t>
void apply_diag(at::Tensor& result, const at::Tensor& self, int64_t dimension) {
  TORCH_CHECK(
      self.dim() == 1 || self.dim() == 2, "matrix or a vector expected");
  int nDimension = self.dim();
  if (nDimension == 2) {
    int sz;
    if (dimension > 0) {
      sz = std::min(self.size(0), self.size(1) - dimension);
    } else {
      sz = std::min(self.size(0) + dimension, self.size(1));
    }
    at::native::resize_output(result, {sz});
    if (sz > 0) {
      at::assert_no_internal_overlap(result);
    }
  } else {
    auto n_elems = self.numel();
    auto sz = (dimension > 0) ? n_elems + dimension : n_elems - dimension;
    at::native::resize_output(result, {sz, sz});
    result.zero_();
    if (sz > 0) {
      at::assert_no_internal_overlap(result);
    }
  }
}

at::Tensor& cnnl_diag_out(
    const at::Tensor& self,
    int64_t dimension,
    at::Tensor& result) {
  AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(self.scalar_type(), "diag", [&] {
    TORCH_CHECK(
        result.scalar_type() == self.scalar_type(),
        "The datatype of out in cnnl_diag_out "
        "must be same as self ",
        self.scalar_type(),
        " but out ",
        result.scalar_type());
    apply_diag<scalar_t>(result, self, dimension);
    auto self_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
    auto result_contiguous =
        cnnl_contiguous(result, self.suggest_memory_format());
    cnnl_diag_internal(result_contiguous, self_contiguous, dimension);
    if (is_copy_necessary(result, result_contiguous)) {
      result.copy_(result_contiguous);
    }
  });
  return result;
}

} // namespace ops
} // namespace torch_mlu
