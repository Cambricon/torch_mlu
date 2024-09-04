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

#include <ATen/native/TensorShape.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::vector<at::Tensor> promote_inputs(const at::TensorList& tensors) {
  // out's dtype is common dtype
  at::ScalarType common_type = tensors[0].scalar_type();
  std::vector<at::Tensor> outputs;
  outputs.emplace_back(tensors[0]);
  // convert input
  for (auto i = 1; i < tensors.size(); ++i) {
    outputs.emplace_back(convertTensorType(tensors[i], common_type));
  }
  return outputs;
}

TORCH_IMPL_FUNC(cat_out_mlu)
(const at::ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 c10::MemoryFormat memory_format,
 const at::Tensor& result) {
  if (result.numel() == 0) {
    return;
  }
  auto materialized = tensors.materialize();
  std::vector<at::Tensor> origins = {result};
  for (const at::Tensor& t : materialized) {
    if (at::native::cat_should_skip_tensor(t))
      continue;
    origins.emplace_back(t);
  }

  // promote dtype of inputs based on result
  auto promote_tensors = promote_inputs(origins);

  // convert memory_format
  std::vector<at::Tensor> contiguous_tensors = {};
  for (const at::Tensor& t : promote_tensors) {
    contiguous_tensors.emplace_back(cnnl_contiguous(t, memory_format));
  }

  // all tensors have same dtype and memory_format
  cnnl_cat_internal(contiguous_tensors, dim, memory_format);

  if (is_copy_necessary(result, contiguous_tensors[0])) {
    result.copy_(contiguous_tensors[0]);
  }
}

} // namespace ops
} // namespace torch_mlu
