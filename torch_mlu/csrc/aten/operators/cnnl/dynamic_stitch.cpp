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

#include <torch/autograd.h>
#include "aten/utils/dispatch.h"
#include "aten/utils/shape_util.h"
#include <ATen/native/Pool.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_dynamic_stitch(
    at::TensorList indices_list,
    at::TensorList data_list) {
  size_t n_tensors = indices_list.size();
  TORCH_CHECK(
      n_tensors == data_list.size(),
      "indices and data must have the same number of tensor.");
  TORCH_CHECK(n_tensors > 0, "Tensor list must have at least one tensor.");
  auto out = at::zeros({0}, data_list[0].options());
  c10::SmallVector<at::Tensor> flatten_indices_list;
  for (const at::Tensor& indices : indices_list) {
    if (indices.numel() > 0) {
      flatten_indices_list.emplace_back(indices.flatten());
    }
  }
  auto all_indices = at::cat(flatten_indices_list);
  int32_t max_index = all_indices.max().item<int32_t>();
  int first_dim_size = max_index + 1;
  const at::Tensor& data0 = data_list[0];
  const at::Tensor& indices0 = indices_list[0];

  int indices_num = 0;
  for (int input_num = 0; input_num < indices_list.size(); input_num++) {
    const at::Tensor& indices = indices_list[input_num];
    const at::Tensor& data = data_list[input_num];
    TORCH_CHECK(
        indices.scalar_type() == at::ScalarType::Int,
        "elements in indices should be int32 but got: ",
        indices.scalar_type());
    TORCH_CHECK(
        indices.is_contiguous(), "indices[", input_num, "] must be contiguous");
    TORCH_CHECK(
        data.is_contiguous(), "data[", input_num, "] must be contiguous");
    TORCH_CHECK(
        ShapeStartsWith(data.sizes(), indices.sizes()),
        "data[",
        input_num,
        "].shape = ",
        data.sizes(),
        " does not start with indices[",
        input_num,
        "].shape = ",
        indices.sizes());
    TORCH_CHECK(
        input_num == 0 || SameExtraShape(data0, indices0, data, indices),
        "Need data[0].shape[",
        indices0.dim(),
        ":] = data[",
        input_num,
        "].shape[",
        indices.dim(),
        ":], got data[0].shape = ",
        data0.sizes(),
        ", data[",
        input_num,
        "].shape = ",
        data.sizes(),
        ", indices[0].shape = ",
        indices0.sizes(),
        ", indices[",
        input_num,
        "].shape = ",
        indices.sizes());

    indices_num += indices.numel();
  }
  auto memory_format = at::MemoryFormat::Contiguous;

  std::vector<int64_t> out_shape = {first_dim_size};
  for (int64_t d = indices0.dim(); d < data0.dim(); ++d) {
    out_shape.push_back(data0.size(d));
  }
  out.resize_(out_shape, memory_format);

  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  if (first_dim_size > 0) {
    AT_DISPATCH_MLU_FLOAT_HALF_AND_INT(
        data0.scalar_type(), "cnnl_dynamic_stitch", [&] {
          cnnl_dynamic_stitch_internal(
              out, indices_list, data_list, indices_num);
        });
  }
  return out;
}

} // namespace ops
} // namespace torch_mlu
