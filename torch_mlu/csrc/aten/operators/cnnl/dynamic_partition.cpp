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

std::vector<at::Tensor> cnnl_dynamic_partition(
    const at::Tensor& data,
    const at::Tensor& partitions,
    int64_t num_partitions) {
  TORCH_CHECK(num_partitions >= 1, "num_partitions must be at least 1");
  TORCH_CHECK(
      partitions.scalar_type() == at::ScalarType::Int,
      "elements in partitions should be int32 but got: ",
      partitions.scalar_type());
  TORCH_CHECK(partitions.is_contiguous(), "partitions must be contiguous");
  TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
  TORCH_CHECK(
      ShapeStartsWith(data.sizes(), partitions.sizes()),
      "data.shape must start with partitions.shape, ",
      "data.shape = ",
      data.sizes(),
      ", partitions.shape = ",
      partitions.sizes());
  std::vector<at::Tensor> outputs;
  if (partitions.numel() == 0) {
    for (int p = 0; p < num_partitions; p++) {
      std::vector<int64_t> out_shape = {0};
      for (int i = partitions.dim(); i < data.dim(); i++) {
        out_shape.push_back(data.size(i));
      }
      auto out = at::empty(out_shape, data.options());
      outputs.emplace_back(out);
    }
    return outputs;
  }
  AT_DISPATCH_MLU_FLOAT_AND_INT(
      data.scalar_type(), "cnnl_dynamic_partition", [&] {
        at::Tensor weight_empty;
        std::vector<int64_t> temp_out_shape = {partitions.numel()};
        for (int i = partitions.dim(); i < data.dim(); i++) {
          temp_out_shape.push_back(data.size(i));
        }
        auto temp_out = at::empty(temp_out_shape, data.options());
        auto out_counts = at::empty({num_partitions}, partitions.options());
        if (data.numel() == 0) {
          cnnl_bincount_internal(
              out_counts,
              partitions,
              weight_empty,
              num_partitions,
              num_partitions);
        } else {
          cnnl_dynamic_partition_internal(
              out_counts, temp_out, data, partitions, num_partitions);
        }
        if ((data.numel() != 0) && (num_partitions == 1)) {
          outputs.emplace_back(temp_out);
        } else {
          auto out_counts_host = out_counts.to(at::Device(at::kCPU));
          for (int p = 0; p < num_partitions; p++) {
            int count = out_counts_host.view(-1)[p].item<int>();
            std::vector<int64_t> out_shape = {count};
            for (int i = partitions.dim(); i < data.dim(); i++) {
              out_shape.push_back(data.size(i));
            }
            auto out = at::empty(out_shape, data.options());
            outputs.emplace_back(out);
          }
          if (data.numel() != 0) {
            cnnl_split_internal(outputs, temp_out, num_partitions, 0);
          }
        }
      });
  return outputs;
}

} // namespace ops
} // namespace torch_mlu
