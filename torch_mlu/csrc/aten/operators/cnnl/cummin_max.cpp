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

#include "ATen/native/LinearAlgebraUtils.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

void cnnl__cummin_helper(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  at::TensorArg input_arg{self, "input", 1};
  at::TensorArg indices_arg{indices, "indices", 2};
  at::TensorArg output_arg{values, "output", 3};
  checkAllSameMLU("cummin", {output_arg, indices_arg, input_arg});
  auto self_contiguous = cast_long_to_int_if_needed(cnnl_contiguous(self));
  auto value_contiguous = create_int_tensor_if_needed(maybe_create_out(
      values,
      values.sizes(),
      get_channels_first_strides(values.sizes()),
      values.options()));

  auto indices_contiguous = cast_long_to_int_if_needed(maybe_create_out(
      indices,
      indices.sizes(),
      get_channels_first_strides(indices.sizes()),
      indices.options()));

  AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(self.scalar_type(), "cummin", [&] {
    cnnl_cummin_max_internal(
        self_contiguous,
        value_contiguous,
        indices_contiguous,
        dim,
        CumType::Cum_Min);
  });
  if (is_copy_necessary(values, value_contiguous)) {
    values.copy_(value_contiguous);
  }
  if (is_copy_necessary(indices, indices_contiguous)) {
    indices.copy_(indices_contiguous);
  }
}

void cnnl__cummax_helper(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  at::TensorArg input_arg{self, "input", 1};
  at::TensorArg indices_arg{indices, "indices", 2};
  at::TensorArg output_arg{values, "output", 3};
  checkAllSameMLU("cummax", {output_arg, indices_arg, input_arg});
  auto self_contiguous = cast_long_to_int_if_needed(cnnl_contiguous(self));
  auto value_contiguous = create_int_tensor_if_needed(maybe_create_out(
      values,
      values.sizes(),
      get_channels_first_strides(values.sizes()),
      values.options()));

  auto indices_contiguous = cast_long_to_int_if_needed(maybe_create_out(
      indices,
      indices.sizes(),
      get_channels_first_strides(indices.sizes()),
      indices.options()));

  AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(self.scalar_type(), "cummax", [&] {
    cnnl_cummin_max_internal(
        self_contiguous,
        value_contiguous,
        indices_contiguous,
        dim,
        CumType::Cum_Max);
  });
  if (is_copy_necessary(values, value_contiguous)) {
    values.copy_(value_contiguous);
  }
  if (is_copy_necessary(indices, indices_contiguous)) {
    indices.copy_(indices_contiguous);
  }
}

} // namespace ops
} // namespace torch_mlu
