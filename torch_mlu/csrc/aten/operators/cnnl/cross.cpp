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
#include <ATen/Dispatch.h>
#include <ATen/native/Cross.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {

using at::infer_size;
using at::maybe_wrap_dim;

TORCH_META_FUNC(linalg_cross_out_mlu)
(const Tensor& input, const Tensor& other, int64_t dim) {
  auto x_d = input.dim();
  auto y_d = other.dim();
  // This is to avoid things like
  // linalg.cross(torch.randn(2, 3), torch.randn(5, 2, 3), dim=2)
  TORCH_CHECK(
      x_d == y_d,
      "linalg.cross: inputs must have the same number of dimensions.");
  TORCH_CHECK(
      input.size(dim) == 3 && other.size(dim) == 3,
      "linalg.cross: inputs dimension ",
      dim,
      " must have length 3. Got ",
      input.size(dim),
      " and ",
      other.size(dim));

  // Broadcast the batch dimension of input and other.
  // Since the non-batch dimensions agree, this is the same as broadcast all the
  // inputs
  auto out_size = infer_size(input.sizes(), other.sizes());

  set_output_raw_strided(
      0,
      out_size,
      {},
      input.options().memory_format(input.suggest_memory_format()));
}

void cross_mlu_kernel(
    const Tensor& result,
    const Tensor& a,
    const Tensor& b,
    const int64_t dim) {
  auto iter = at::TensorIteratorConfig()
                  .add_output(result)
                  .add_input(a)
                  .add_input(b)
                  .resize_outputs(false)
                  .declare_static_shape(result.sizes(), /*squash_dims=*/dim)
                  .build();

  if (iter.numel() == 0) {
    return;
  }

  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "cross");
  auto output = iter.output(0);
  output = create_int_tensor_if_needed(output);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(
      at::kHalf, iter.common_dtype(), "cross_mlu_kernel", [&] {
        cnnl_cross_internal(
            output,
            cast_long_to_int_if_needed(iter.input(0)),
            cast_long_to_int_if_needed(iter.input(1)),
            dim);
      });

  cast_int_to_long_if_needed(output, iter.output(0));
  iter_bridge.cast_outputs(iter);
}

TORCH_IMPL_FUNC(linalg_cross_out_mlu)
(const Tensor& input, const Tensor& other, int64_t dim, const Tensor& out) {
  dim = maybe_wrap_dim(dim, input.dim());
  auto out_size = out.sizes();
  Tensor input_broadcasted = input.expand(out_size);
  Tensor other_broadcasted = other.expand(out_size);
  cross_mlu_kernel(out, input_broadcasted, other_broadcasted, dim);
}

} // namespace ops
} // namespace torch_mlu
