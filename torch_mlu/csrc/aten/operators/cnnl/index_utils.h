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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

struct MLUAdvancedIndex {
  MLUAdvancedIndex(
      const at::Tensor& src,
      at::TensorList indices_list,
      bool hasContiguousSubspace,
      std::vector<bool> hasDefined);
  at::Tensor self;
  bool hasContiguousSubspace;
  at::Tensor src;
  std::vector<at::Tensor> indices;
  std::vector<bool> hasDefined;
  at::DimVector indexed_sizes;
  at::DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

MLUAdvancedIndex make_info(
    at::Tensor self,
    const c10::List<std::optional<at::Tensor>>& orig);

// This function is used by make_info in index.cpp, which is used to check
// whether data type of indices are in byte, bool and long.
static void checkIndexTensorTypes(
    const torch::List<std::optional<at::Tensor>>& indices) {
  for (std::optional<at::Tensor> tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      auto scalarType = tensor->scalar_type();
      if (scalarType != at::ScalarType::Long &&
          scalarType != at::ScalarType::Byte &&
          scalarType != at::ScalarType::Bool &&
          scalarType != at::ScalarType::Int) {
        TORCH_CHECK_INDEX(
            false,
            "tensors used as indices must be long, byte or bool tensors");
      }
    }
  }
}

std::vector<at::Tensor> expand_indices(
    const at::Tensor& self,
    std::vector<at::Tensor> indices,
    const int64_t& indice_dim,
    const std::vector<int64_t>& output_dims,
    const std::vector<int64_t>& def_reshape_size);

// In make_info, when empty indice(s) are used, a native function called
// transposeToFront() will be called. The function transposeToFront will
// transpose the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor and
// the reordered indices. For example: transposeToFront(tensor, {nullptr, a,
// nullptr, b}) returns tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
// In the native function, info.src will be passed into corresponding CPU kernel
// and CUDA kernel. Info.src is all the elements which will be involved in
// indexing operations in self tensor. However, cnnlIndexPut kernel takes the
// whole permuted self tensor. Therefore, we use this function to calculate the
// desired ordering of dimensions so that the permuted self tensor will have the
// same shape with the input tensor.
static std::vector<int64_t> transposed_dims(
    at::Tensor self,
    std::vector<bool> hasDefined) {
  std::vector<int64_t> dims, dims_back;
  dims.reserve(self.dim());
  dims_back.reserve(self.dim());
  // The following 2 for loops are used to calculate the dims used by
  // at::native::transposeToFront
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (hasDefined[i]) {
      dims.push_back(i);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!hasDefined[i]) {
      dims.push_back(i);
    }
  }
  // The following code is used to calculate the dims, where dims[i] =
  // position(i). For example, dims vector used by transposeToFront is [0, 2, 4,
  // 1, 3], the desired dims are [0, 3, 1, 4, 2]
  for (auto i = 0; i < dims.size(); i++) {
    std::vector<int64_t>::iterator itr = std::find(dims.begin(), dims.end(), i);
    dims_back.push_back(std::distance(dims.begin(), itr));
  }
  return dims_back;
}
} // namespace ops
} // namespace torch_mlu
