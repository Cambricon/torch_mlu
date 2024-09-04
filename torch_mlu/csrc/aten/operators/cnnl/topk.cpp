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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

TORCH_IMPL_FUNC(topk_out_mlu)
(const at::Tensor& self,
 int64_t k,
 int64_t dim,
 bool largest,
 bool sorted,
 const at::Tensor& values,
 const at::Tensor& indices) {
  at::TensorArg topK_arg{values, "topK", 1}, indices_arg{indices, "indices", 2},
      input_arg{self, "self", 3};
  checkAllSameMLU(__func__, {topK_arg, indices_arg, input_arg});

  dim = at::maybe_wrap_dim(dim, self);

  // If k is 0 the result is an empty tensor, so we don't need to launch a
  // kernel.
  if (k == 0) {
    return;
  }

  if (self.numel() == 0) {
    return;
  }

  auto memory_format = self.suggest_memory_format();
  auto self_contiguous =
      cast_long_to_int_if_needed(cnnl_contiguous(self, memory_format));
  auto values_contiguous =
      create_int_tensor_if_needed(cnnl_contiguous(values, memory_format));
  bool indices_support_long = topk_indices_support_long(self.scalar_type());
  at::Tensor indices_contiguous;
  if (indices_support_long) {
    indices_contiguous = cnnl_contiguous(indices, memory_format);
  } else {
    indices_contiguous =
        create_int_tensor_if_needed(cnnl_contiguous(indices, memory_format));
  }

  cnnl_topk_internal(
      values_contiguous,
      indices_contiguous,
      self_contiguous,
      k,
      dim,
      largest,
      sorted,
      /*stable=*/true);

  if (is_copy_necessary(values, values_contiguous)) {
    values.copy_(values_contiguous);
  }
  if (is_copy_necessary(indices, indices_contiguous)) {
    indices.copy_(indices_contiguous);
  }
}

} // namespace ops
} // namespace torch_mlu
