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
#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

// Using cnnl_embedding to improve perf
at::Tensor cnnl_embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  TORCH_CHECK(weight.dim() == 2, "'weight' must be 2-D");
  auto indices_arg = at::TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding", indices_arg, {at::kLong, at::kInt});

  TORCH_MLU_CHECK(
      at::isFloatingType(weight.scalar_type()),
      "embedding_forward expected weight to have floating type, got ",
      weight.scalar_type());

  auto size = indices.sizes().vec();
  if (indices.dim() == 0 && indices.numel() == 1) {
    size = std::vector<int64_t>{1};
  }
  for (auto d : weight.sizes().slice(1)) {
    size.push_back(d);
  }
  auto memory_format = weight.suggest_memory_format();
  auto output = at::empty(size, weight.options().memory_format(memory_format));
  if (indices.numel() == 0) {
    return output;
  }

  auto weight_contiguous = cnnl_contiguous(weight, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);
  // padding_idx deal with python func
  // see
  // https://pytorch.org/docs/1.13/_modules/torch/nn/modules/sparse.html#Embedding
  // for more details
  cnnl_embedding_internal(
      weight_contiguous, indices_contiguous, output, (-1), scale_grad_by_freq);
  if (indices.dim() == 0 && indices.numel() == 1) {
    output.squeeze_(0);
  }
  return output;
}

} // namespace ops
} // namespace torch_mlu
