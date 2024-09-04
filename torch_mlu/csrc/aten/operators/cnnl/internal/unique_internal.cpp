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

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_unique_internal(
    const at::Tensor& self,
    int64_t dim,
    bool return_inverse,
    bool return_counts) {
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor self_desc;
  CnnlUniqueDescriptor unique_desc;

  if (dim != -1) {
    dim = at::maybe_wrap_dim(dim, self.dim());
  }
  // When input data type is int64,
  // CNNL only supports the unique of the flattened input,
  // the dim is set to -1, or dim is set to 0 if the input is one dimensional
  bool need_cast = false;
  at::Tensor input = self;
  if (self.scalar_type() == at::kLong && (dim >= 0 && self.dim() != 1)) {
    input = self.to(at::kInt);
    need_cast = true;
  }
  auto self_impl = getMluTensorImpl(input);
  self_desc.set(input);
  auto self_ptr = self_impl->mlu_data_ptr();

  // The current MLU implementation of unique always sort
  unique_desc.set(true /*sorted*/, dim, return_inverse, return_counts);

  // get workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetUniqueWorkspaceSize(
      handle, unique_desc.desc(), self_desc.desc(), &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  at::Tensor output_len;
  void* output_len_ptr = nullptr;
  output_len = at::empty({1}, self.options().dtype(at::ScalarType::Int));
  output_len_ptr = getMluTensorImpl(output_len)->mlu_data_ptr();

  // output
  at::Tensor out_data;
  void* out_data_ptr = nullptr;
  // the memory we allocate is greater or equal to actual size
  if (dim != -1) {
    out_data = at::empty(self.sizes(), self.options());
  } else {
    out_data = at::empty({self.numel()}, self.options());
  }
  auto output = out_data;
  // When input data type is int64,
  // CNNL only supports the unique of the flattened input,
  // the dim is set to -1, or dim is set to 0 if the input is one dimensional
  if (need_cast) {
    output = at::empty_like(out_data, out_data.options().dtype(at::kInt));
  }
  out_data_ptr = getMluTensorImpl(output)->mlu_data_ptr();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  // indices
  at::Tensor out_data_index =
      at::empty({0}, self.options().dtype(at::ScalarType::Long));
  void* out_data_index_ptr = nullptr;
  if (return_inverse) {
    // when dim is specified, the tensor indices is one-dimensional
    // and the size of dimension is equal to the size of input tensor
    if (dim != -1) {
      out_data_index =
          at::empty(self.size(dim), self.options().dtype(at::ScalarType::Int));
    } else {
      // when dim is set to -1, the tensor indices is same
      // shape as input tensor
      out_data_index =
          at::empty_like(self, self.options().dtype(at::ScalarType::Int));
    }
    out_data_index_ptr = getMluTensorImpl(out_data_index)->mlu_data_ptr();
  }
  CnnlTensorDescriptor indices_desc;
  indices_desc.set(out_data_index);

  // count
  at::Tensor out_counts =
      at::empty({0}, self.options().dtype(at::ScalarType::Long));
  void* out_counts_ptr = nullptr;
  if (return_counts) {
    // the tensor counts is one-dimensional when dim is specified
    if (dim != -1) {
      out_counts = at::empty(
          out_data.size(dim), out_data.options().dtype(at::ScalarType::Int));
    } else {
      // the tensor counts is same shape with output when dim is -1
      out_counts = at::empty_like(
          out_data, out_data.options().dtype(at::ScalarType::Int));
    }
    out_counts_ptr = getMluTensorImpl(out_counts)->mlu_data_ptr();
  }
  CnnlTensorDescriptor counts_desc;
  counts_desc.set(out_counts);

  TORCH_CNNL_CHECK(cnnlUnique_v2(
      handle,
      unique_desc.desc(),
      self_desc.desc(),
      self_ptr,
      workspace_ptr.get(),
      workspace_size,
      static_cast<int*>(output_len_ptr),
      output_desc.desc(),
      out_data_ptr,
      indices_desc.desc(),
      out_data_index_ptr,
      counts_desc.desc(),
      out_counts_ptr));
  if (need_cast) {
    out_data = output.to(at::kLong);
  }

  int output_num = output_len.item<int>();
  if (dim != -1) {
    std::vector<int64_t> output_size;
    for (auto& s : self.sizes()) {
      output_size.push_back(s);
    }
    output_size[dim] = output_num;
    out_data.resize_(output_size);
  } else {
    out_data.resize_(output_num);
  }
  out_counts.resize_(output_num);
  return std::make_tuple(
      out_data, out_data_index.to(at::kLong), out_counts.to(at::kLong));
}

} // namespace ops
} // namespace torch_mlu
