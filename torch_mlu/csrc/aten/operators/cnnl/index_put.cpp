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
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/ExpandUtils.h>
#include "ATen/native/IndexingUtils.h"
#include "aten/operators/cnnl/index_utils.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// This function is used to decide whether an index_put op can be dispatched to
// masked_fill op. In native PyTorch, a similar function is implemented in
// pytorch/aten/src/ATen/native/TensorAdvancedIndexingUtils.h. However,
// canDispatchToMaskedFill() in native PyTorch cannot be reused since value
// tensor is not on MLU.

static std::tuple<bool, at::Tensor> canDispatchToMaskedFill(
    const at::Tensor self,
    const torch::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& value) {
  if (!(value.numel() == 1 &&
        (value.device().is_privateuseone() || value.device().is_cpu()))) {
    return std::make_tuple(false, Tensor());
  }
  int64_t num_ind = 0;
  at::Tensor mask;
  auto self_device = self.device();
  for (const std::optional<Tensor> i : indices) {
    if (!i.has_value() || !(*i).defined()) {
      num_ind++;
    } else {
      at::Tensor index = std::move(*i);
      if ((index.scalar_type() != at::kByte &&
           index.scalar_type() != at::kBool) ||
          index.device() != self_device || mask.defined()) {
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(
              index.size(j) == self.size(srcIdx),
              "The shape of the mask ",
              index.sizes(),
              " at index ",
              j,
              " does not match the shape of the indexed tensor ",
              self.sizes(),
              " at index ",
              srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for (const auto i : c10::irange(num_ind, self.ndimension())) {
    (void)i; // Suppress unused variable warning
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}

at::Tensor& cnnl__index_put_impl_(
    at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values,
    const bool accumulate,
    const bool unsafe) {
  // see PYTORCH-12482
  TORCH_CHECK(
      self.device().is_privateuseone(),
      "Currently, index_put on MLU does not support input that is not on MLU, now the input is on ",
      self.device(),
      ".")
  TORCH_CHECK(
      indices.size() <= self.dim(), "indices have more indices than self dim");
  TORCH_CHECK(!indices.empty(), "indices can't be empty");
  // TODO(huangqipeng): index_put_ of mlu not support self_overlap,
  // because self_overlap.copy_(self_contiguous) will be error.
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_put_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, values);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), values);
    }
  }
  auto info = make_info(self, indices);
  auto self_expanded = info.self;
  std::vector<at::Tensor> indices_expand;
  TORCH_CHECK(
      at::is_expandable_to(values.sizes(), info.src.sizes()),
      "shape mismatch: value tensor of shape ",
      values.sizes(),
      " cannot be broadcast to indexing result of shape ",
      info.src.sizes());
  TORCH_CHECK(
      values.scalar_type() == info.src.scalar_type(),
      "Index put requires the source and destination dtypes match, "
      "got ",
      info.src.scalar_type(),
      " for the destination "
      "and ",
      values.scalar_type(),
      " for the source.");
  if (self.numel() == 0) {
    return self;
  }
  if (indices.size() == 1 && (*indices[0]).defined() &&
      indices[0]->scalar_type() == at::ScalarType::Bool) {
    indices_expand.emplace_back(indices[0].value());
  } else {
    indices_expand = info.indices;
  }
  auto values_ = values;
  if (values.device() != self.device() && values.numel() == 1 &&
      values.dim() == 0) {
    values_ = values.to(self.device());
  }
  at::assert_no_overlap(self, values);
  for (const std::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }
  std::vector<at::Tensor> indices_expand_;
  for (const auto& indice : indices_expand) {
    // only support long, int32 and bool
    auto indice_ = indice;
    if (!indice.defined()) {
      indices_expand_.push_back(indice_);
      continue;
    }
    TORCH_CHECK(
        indice.scalar_type() == at::ScalarType::Int ||
            indice.scalar_type() == at::ScalarType::Bool ||
            indice.scalar_type() == at::ScalarType::Long,
        "support only int, bool and long");
    if (indice.device() != self.device()) {
      indice_ = indice.to(self.device());
    }
    indices_expand_.push_back(indice_);
  }
  auto self_contiguous =
      cnnl_contiguous(self_expanded, c10::MemoryFormat::Contiguous);
  auto value_contiguous =
      cnnl_contiguous(values_, c10::MemoryFormat::Contiguous);

  // If accumulate is false, input/values/output support the following data
  // widths: 1-byte, 2-byte, 4-byte, 8-byte. If accumulate is true,
  // input/values/output support the following data type: half, bfloat16, float,
  // int32, int16, int8, uint8.
  if (accumulate) {
    self_contiguous = cast_long_to_int_if_needed(self_contiguous);
    value_contiguous = cast_long_to_int_if_needed(value_contiguous);
  }

  cnnl_index_put_internal(
      self_contiguous,
      self_contiguous,
      indices_expand_,
      value_contiguous,
      accumulate);
  if (!info.hasContiguousSubspace) {
    auto permute_dims = transposed_dims(self, info.hasDefined);
    self_contiguous = self_contiguous.permute(permute_dims);
  }
  if (is_copy_necessary(self, self_contiguous)) {
    self.copy_(self_contiguous);
  }

  return self;
}

} // namespace ops
} // namespace torch_mlu
