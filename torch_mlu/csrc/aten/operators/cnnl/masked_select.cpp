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
#include "ATen/ExpandUtils.h"
#include "ATen/NamedTensorUtils.h"
#include "ATen/core/NamedTensor.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// Type list supported by GPU.
// coda path: aten/src/ATen/native/cuda/IndexKernel.cpp
// masked type list:       at::ScalarType::Byte, at::ScalarType::Bool
// input/output type list: at::ScalarType::Byte, at::ScalarType::Char,
//                         at::ScalarType::Int, at::ScalarType::Long,
//                         at::ScalarType::Short, at::ScalarType::Double,
//                         at::ScalarType::Float, at::ScalarType::ComplexDouble,
//                         at::ScalarType::ComplexFloat,
//                         at::ScalarType::ComplexHalf, at::ScalarType::Half,
//                         at::ScalarType::Bool, at::ScalarType::BFloat16.

// Type list supported by CNNL masked select.
// masked type list:       at::ScalarType::Byte, at::ScalarType::Bool
// input/output type list: at::ScalarType::Byte, at::ScalarType::Char,
//                         at::ScalarType::Short, at::ScalarType::Int,
//                         at::ScalarType::Half, at::ScalarType::Bool,
//                         at::ScalarType::Float, at::ScalarType::Long
//                         at::ScalarType::Double
// Not real support double, but support double to float convert.

at::Tensor cnnl_masked_select(const at::Tensor& self, const at::Tensor& mask) {
  at::namedinference::compute_broadcast_outnames(self, mask);
  auto out = at::empty({0}, self.options());
  return cnnl_masked_select_out(self, mask, out);
}

at::Tensor& cnnl_masked_select_out(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& output) {
  at::namedinference::compute_broadcast_outnames(self, mask);
  at::NoNamesGuard guard;
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte ||
          mask.scalar_type() == at::ScalarType::Bool,
      "masked_select: expected BoolTensor or ByteTensor for mask");
  TORCH_CHECK(
      self.scalar_type() == output.scalar_type(),
      "masked_select(): self and output must have the same scalar type");
  auto mask_temp = (mask.dim() == 0)
      ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
      : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
      ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
      : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = at::expand_outplace(*mask_temp, *self_temp);
  cnnl_index_out(
      *std::get<1>(mask_self_expanded),
      c10::List<std::optional<at::Tensor>>(
          {*std::move(std::get<0>(mask_self_expanded))}),
      output);

  return output;
}

} // namespace ops
} // namespace torch_mlu
