/*
All modification made by Cambricon Corporation: © 2024 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2024, the respective contributors
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

#include "aten/utils/shape_util.h"

namespace torch_mlu::ops {

bool ShapeStartsWith(
    const at::IntArrayRef& shape,
    const at::IntArrayRef& prefix) {
  if (shape.size() < prefix.size()) {
    return false;
  }

  for (size_t i = 0; i < prefix.size(); ++i) {
    if (shape[i] != prefix[i]) {
      return false;
    }
  }
  return true;
}

// Check if data0.shape[indices0.dims():] == data1.shape[indices1.dims():]
bool SameExtraShape(
    const Tensor& data0,
    const Tensor& indices0,
    const Tensor& data1,
    const Tensor& indices1) {
  // Calculate "extra dimensions"
  int64_t extra0 = data0.dim() - indices0.dim();
  int64_t extra1 = data1.dim() - indices1.dim();

  // If the number of extra dimensions is different, return false
  if (extra0 != extra1) {
    return false;
  }
  if (extra0 == 0) {
    return true;
  }
  // Compare the sizes of the extra dimensions
  for (int64_t i = 0; i < extra0; ++i) {
    if (data0.size(indices0.dim() + i) != data1.size(indices1.dim() + i)) {
      return false;
    }
  }
  return true;
}

} // namespace torch_mlu::ops
