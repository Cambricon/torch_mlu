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

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_embedding_dense_backward(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  TORCH_CHECK(
      at::isFloatingType(grad_output.scalar_type()),
      "embedding_dense_backward expected tensor to have floating type, got ",
      grad_output.scalar_type());
  auto grad_arg = at::TensorArg(grad_output, "grad_output", 1);
  auto indices_arg = at::TensorArg(indices, "indices", 1);
  at::checkScalarTypes(
      "embedding_backward", indices_arg, {at::kLong, at::kInt});
  torch_mlu::checkSameMLU("embedding_backward", grad_arg, indices_arg);
  if (indices.defined() && indices.numel() == 0) {
    std::vector<int64_t> out_vec = {num_weights, grad_output.size(-1)};
    at::IntList out_shape(out_vec);
    auto output_ = at::zeros(out_shape, grad_output.options());
    return output_;
  }
  auto memory_format =
      switch_tensors_suggest_memory_format({grad_output, indices});
  auto grad_contig = cnnl_contiguous(grad_output, memory_format);
  auto indices_contig = cnnl_contiguous(indices, memory_format);
  auto output =
      at::empty({num_weights, grad_output.size(-1)}, grad_contig.options());
  cnnl_embedding_dense_backward_internal(
      grad_contig,
      indices_contig.to(at::kInt),
      num_weights,
      padding_idx,
      scale_grad_by_freq,
      output);
  return output;
}
} // namespace ops
} // namespace torch_mlu
