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

#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/scatter_utils.h"

namespace torch_mlu {
namespace ops {

using at::assert_no_internal_overlap;
using at::native::gather_stub;

std::set<at::ScalarType> gather_support_dtype{
    at::ScalarType::Int,
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    at::ScalarType::Float,
    at::ScalarType::Short,
    at::ScalarType::Char,
    at::ScalarType::Byte,
    at::ScalarType::Bool,
    at::ScalarType::Long,
    at::ScalarType::Double};

void gather_mlu_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  TORCH_CHECK(
      gather_support_dtype.find(self.scalar_type()) !=
          gather_support_dtype.end(),
      "gather mlu op not implemented for '",
      self.scalar_type(),
      "'");
  assert_no_internal_overlap(result);
  auto self_contiguous = cnnl_contiguous(self);
  auto result_contiguous = cnnl_contiguous(result);

  at::Tensor index_internal = index;
  uint8_t cnnl_limit = 64;
  bool stride_index_flag =
      canHandleStrideScatterGatherIndex(index, dim, cnnl_limit);
  if (!stride_index_flag) {
    index_internal = cnnl_contiguous(index, at::MemoryFormat::Contiguous);
  }

  cnnl_gather_internal(result_contiguous, self_contiguous, dim, index_internal);
  if (!result.is_same(result_contiguous))
    result.copy_(result_contiguous);
}

REGISTER_PRIVATEUSE1_DISPATCH(gather_stub, &gather_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
