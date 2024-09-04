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
#include "ATen/native/TensorAdvancedIndexing.h"
#include "ATen/native/IndexKernel.h"
#include "aten/TensorIteratorBridge.h"
#include <ATen/MemoryOverlap.h>
#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_index_copy(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  at::Tensor output = at::empty_like(self);
  return cnnl_index_copy_out(self, dim, index, source, output);
}

at::Tensor& cnnl_index_copy_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& out) {
  dim = c10::maybe_wrap_dim(dim, self.dim());
  bool check_result = out.defined();
  if (check_result) {
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, index);
    at::assert_no_overlap(out, source);
  }
  TORCH_CHECK_INDEX(
      index.dim() < 2,
      "index_copy_(): Index should have dimension 1 or 0 (got ",
      index.dim(),
      ")");
  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source is scalar,\
                          index should have one element (got ",
        numIndices,
        ")");
  } else if (
      (source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source and destination are not scalars,\
                          their dimensionality must match. Source dimensionality (",
        source.dim(),
        "), destination dimensionality (",
        self.dim(),
        ")");
  }
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Long,
      "index_copy_(): Expected a long tensor for index, but got ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "index_copy_(): self and source expected to have the same dtype, but got (self) ",
      self.scalar_type(),
      " and (source) ",
      source.scalar_type());
  TORCH_CHECK(
      self.device() == source.device() && self.device() == index.device(),
      "index_copy_(): self, index and source expected to be in the same device, but got (self) ",
      self.device(),
      ", (index) ",
      index.device(),
      ", and (source) ",
      source.device());
  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (selfSlicedSizes.size() > 0) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (sourceSlicedSizes.size() > 0) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(
          selfSlicedSizes.begin(),
          selfSlicedSizes.end(),
          sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension "
       << dim;
    ss << " and source slice shape: " << sourceSlicedSizes
       << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(
      source.dim() == 0 || numIndices == source.size(dim),
      "index_copy_(): Number of indices (",
      numIndices,
      ") should be equal to source.size(dim) (",
      source.size(dim),
      ")");
  TORCH_CHECK(
      !c10::isComplexType(self.scalar_type()),
      "Complex type input is not supported yet!");
  if (!out.is_same(self)) {
    out.resize_as_(self).copy_(self);
  }
  if (out.numel() == 0 || index.numel() == 0)
    return out;
  if (out.dim() == 0) {
    out.fill_(source.item());
    return out;
  }
  // TODO(hyl): The reason of inde_copy not to use Tensoriterator:
  // index_copy of mlu not support self_overlap,
  // because self_overlap.copy_(self_contiguous) will be error.
  // but iter config contains set_check_mem_overlap(false)
  auto out_contiguous = cnnl_contiguous(out);
  auto self_contiguous = cnnl_contiguous(self);
  auto index_contiguous = cnnl_contiguous(index);
  auto source_contiguous = cnnl_contiguous(source);
  dim = at::maybe_wrap_dim(dim, self);
  cnnl_index_copy_internal(
      out_contiguous,
      self_contiguous,
      dim,
      index_contiguous,
      source_contiguous);
  if (is_copy_necessary(out, out_contiguous)) {
    out.copy_(out_contiguous);
  }
  return out;
}

at::Tensor& cnnl_index_copy_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  return cnnl_index_copy_out(self, dim, index, source, self);
}

} // namespace ops
} // namespace torch_mlu
