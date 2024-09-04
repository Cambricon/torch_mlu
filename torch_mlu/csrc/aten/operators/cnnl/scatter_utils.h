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

#pragma once

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// TODO(mengpenghui): cnnlScatter can handle all case for index with stride in
// the future. Now, cnnlScatter can handle index with stride under the following
// conditions.
// 1. The stride of the index is formed by expand op, and must low dimension 1
// -> n,
//    that is, the stride is (A, B, C, 0, 0, 0)
// 2. the part of index size <= dim shoule contigous or
//    product of the part of index size > dim, should >= cnnl_limit and < 40960
// 3. The dim parameter cannot be the last dimension
inline bool canHandleStrideScatterGatherIndex(
    const at::Tensor& index,
    int64_t dim,
    uint8_t cnnl_limit) {
  if (index.is_contiguous()) {
    return false;
  }

  auto dim_size = index.dim();
  bool condition_c = (dim_size - 1 != dim);
  if (!condition_c)
    return false;

  auto strides = index.strides().vec();
  auto sizes = index.sizes().vec();
  bool condition_a = (strides[dim_size - 1] == 0);
  if (!condition_a)
    return false;

  int64_t end_shape_mul = 1;
  for (auto i = dim_size - 1; i > dim; --i) {
    if (strides[i] == 0) {
      end_shape_mul *= sizes[i];
    } else {
      break;
    }
  }
  bool condition_b = (end_shape_mul >= cnnl_limit && end_shape_mul < 40960);
  return condition_b;
}

} // namespace ops
} // namespace torch_mlu
