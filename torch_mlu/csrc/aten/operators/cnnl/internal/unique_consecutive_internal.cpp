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

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_unique_consecutive_internal(
    const at::Tensor& self,
    int64_t dim,
    bool return_inverse,
    bool return_counts) {
  // [in] handle
  auto handle = getCurrentHandle();

  // [in] unique_consecutive_desc
  if (dim != self.dim())
    dim = at::maybe_wrap_dim(dim, self.dim());
  CnnlUniqueConsecutiveDescriptor unique_consecutive_desc;
  unique_consecutive_desc.set(dim, return_inverse, return_counts);

  // [in] self_desc & [in] self
  CnnlTensorDescriptor self_desc;
  self_desc.set(self);
  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = self_impl->mlu_data_ptr();

  // [in] workspace & [in] workspace_size
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetUniqueConsecutiveWorkspaceSize(
      handle,
      unique_consecutive_desc.desc(),
      self_desc.desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // [out] output_num
  at::Tensor output_num;
  void* output_num_ptr = nullptr;
  output_num = at::empty({1}, self.options().dtype(at::ScalarType::Long));
  output_num_ptr = getMluTensorImpl(output_num)->mlu_data_ptr();

  // [in] output_desc & [out] output
  at::Tensor output;
  void* output_ptr = nullptr;
  output = at::empty(self.sizes(), self.options());
  output_ptr = getMluTensorImpl(output)->mlu_data_ptr();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  // [in] indices_desc & [out] inverse_indices
  at::Tensor inverse_indices =
      at::empty({0}, self.options().dtype(at::ScalarType::Long));
  void* inverse_indices_ptr = nullptr;
  if (return_inverse) {
    if (dim != self.dim()) { // dim != None
      inverse_indices =
          at::empty(self.size(dim), self.options().dtype(at::ScalarType::Long));
    } else { // dim == None
      inverse_indices =
          at::empty_like(self, self.options().dtype(at::ScalarType::Long));
    }
    inverse_indices_ptr = getMluTensorImpl(inverse_indices)->mlu_data_ptr();
  }
  CnnlTensorDescriptor indices_desc;
  indices_desc.set(inverse_indices);

  // [in] counts_desc & [out] counts
  at::Tensor counts =
      at::empty({0}, self.options().dtype(at::ScalarType::Long));
  void* counts_ptr = nullptr;
  if (return_counts) {
    if (dim != self.dim()) { // dim != None
      counts = at::empty(
          output.size(dim), output.options().dtype(at::ScalarType::Long));
    } else { // dim == None
      counts =
          at::empty_like(output, output.options().dtype(at::ScalarType::Long));
    }
    counts_ptr = getMluTensorImpl(counts)->mlu_data_ptr();
  }
  CnnlTensorDescriptor counts_desc;
  counts_desc.set(counts);

  // unique_consecutive operation
  TORCH_CNNL_CHECK(cnnlUniqueConsecutive(
      handle,
      unique_consecutive_desc.desc(),
      self_desc.desc(),
      self_ptr,
      workspace_ptr.get(),
      workspace_size,
      static_cast<int64_t*>(output_num_ptr),
      output_desc.desc(),
      output_ptr,
      indices_desc.desc(),
      inverse_indices_ptr,
      counts_desc.desc(),
      counts_ptr));

  // post-processing: reshape output and counts according to output_num
  int64_t total_elements = *static_cast<int64_t*>(output_num.cpu().data_ptr());
  if (dim != self.dim()) { // dim != None
    auto output_size = self.sizes().vec();
    output_size[dim] = total_elements * self.sizes().vec()[dim] / self.numel();
    output.resize_(output_size);
    counts.resize_(output_size[dim]);
  } else { // dim == None
    output.resize_(total_elements);
    counts.resize_(total_elements);
  }

  return std::make_tuple(output, inverse_indices, counts);
}

} // namespace ops
} // namespace torch_mlu
