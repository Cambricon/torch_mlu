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

#include "ATen/native/Sorting.h"
#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

using at::native::sort_stub;
std::set<at::ScalarType> sort_support_types{
    at::kFloat,
    at::kDouble,
    at::kInt,
    at::kLong,
    at::kShort,
    at::kHalf,
    at::kBFloat16,
    at::kChar,
    at::kByte};
void sort_mlu_kernel(
    const at::TensorBase& self_base,
    const at::TensorBase& values_base,
    const at::TensorBase& indices_base,
    int64_t dim,
    bool descending,
    bool stable) {
#define TOTENSOR(BASE, VAR)               \
  at::OptionalTensorRef opt_##BASE(BASE); \
  const at::Tensor& VAR = *opt_##BASE;

  TOTENSOR(self_base, self);
  TOTENSOR(values_base, values);
  TOTENSOR(indices_base, indices);

  auto self_dtype = self.scalar_type();
  TORCH_CHECK(
      sort_support_types.find(self_dtype) != sort_support_types.end(),
      "Sort currently does not support ",
      self_dtype,
      " dtypes on MLU.");

  bool indices_support_long = topk_indices_support_long(self.scalar_type());

  if (self.numel() == 0)
    return;
  auto memory_format = self.suggest_memory_format();
  auto self_contiguous =
      cast_long_to_int_if_needed(cnnl_contiguous(self, memory_format));
  auto values_contiguous =
      create_int_tensor_if_needed(cnnl_contiguous(values, memory_format));
  at::Tensor indices_contiguous;
  if (indices_support_long) {
    indices_contiguous = cnnl_contiguous(indices, memory_format);
  } else {
    indices_contiguous =
        create_int_tensor_if_needed(cnnl_contiguous(indices, memory_format));
  }
  int64_t k = self.size(dim);
  cnnl_topk_internal(
      values_contiguous,
      indices_contiguous,
      self_contiguous,
      k,
      dim,
      descending,
      /*sorted*/ true,
      stable);
  if (!values.is_same(values_contiguous))
    values.copy_(values_contiguous);
  if (!indices.is_same(indices_contiguous))
    indices.copy_(indices_contiguous);
}

REGISTER_PRIVATEUSE1_DISPATCH(sort_stub, &sort_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
