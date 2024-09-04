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
#include "ATen/native/ReduceOpsUtils.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_unfold(
    const at::Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

at::Tensor cnnl_unfold_backward(
    const at::Tensor& grad_out,
    at::IntArrayRef input_sizes,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad_out.options());
  if (grad_input.numel() == 0) {
    return grad_input;
  }
  int64_t input_dim = input_sizes.size();
  auto dim = at::maybe_wrap_dim(dimension, input_dim);
  std::vector<int64_t> stride;
  // fake stride for contiguous input
  int64_t z = 1;
  stride.insert(stride.begin(), z);
  for (int64_t d = input_dim - 1; d > 0; d--) {
    z *= input_sizes[d];
    stride.insert(stride.begin(), z);
  }

  std::vector<int64_t> new_stride(input_dim + 1);
  new_stride[input_dim] = input_dim == 0 ? 1 : stride[dim];
  for (int64_t d = 0; d < input_dim; d++) {
    auto input_stride = stride[d];
    if (d == dim) {
      new_stride[d] = step * input_stride;
    } else {
      new_stride[d] = input_stride;
    }
  }

  auto grad_contiguous = cnnl_contiguous(grad_out);

  return cnnl_as_strided_backward_internal(
      grad_input, grad_contiguous, new_stride, 0);
}

} // namespace ops
} // namespace torch_mlu
