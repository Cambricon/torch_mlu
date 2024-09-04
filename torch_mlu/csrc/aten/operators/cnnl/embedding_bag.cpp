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
#include <ATen/TensorUtils.h>

namespace torch_mlu {
namespace ops {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

std::pair<at::Tensor, at::Tensor> promoteIndicesAndOffsets(
    const at::Tensor& indices,
    const at::Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__embedding_bag_forward_only(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const std::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx) {
  return cnnl__embedding_bag(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cnnl__embedding_bag(
    const at::Tensor& weight_,
    const at::Tensor& indices_,
    const at::Tensor& offsets_,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const std::optional<at::Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const at::Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  auto per_sample_weights_contiguous = per_sample_weights.defined()
      ? cnnl_contiguous(per_sample_weights)
      : per_sample_weights;
  auto weight = cnnl_contiguous(weight_);

  at::Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  auto indices_arg = at::TensorArg(indices, "indices", 1);
  at::checkScalarTypes(
      "cnnl__embedding_bag", indices_arg, {at::kLong, at::kInt});
  auto offsets_arg = at::TensorArg(offsets, "offsets", 1);
  at::checkScalarTypes(
      "cnnl__embedding_bag", offsets_arg, {at::kLong, at::kInt});
  at::checkSameType("cnnl__embedding_bag", indices_arg, offsets_arg);
  auto weight_arg = at::TensorArg(weight, "weight", 1);
  torch_mlu::checkSameMLU("cnnl__embedding_bag", weight_arg, indices_arg);
  torch_mlu::checkSameMLU("cnnl__embedding_bag", weight_arg, offsets_arg);
  if (per_sample_weights_contiguous.defined()) {
    auto per_sample_weights_arg = at::TensorArg(
        per_sample_weights_contiguous, "per_sample_weights_contiguous", 1);
    TORCH_CHECK(
        weight_arg->options().type_equal(per_sample_weights_arg->options()),
        "expected scalar type ",
        weight_arg->toString(),
        " but found ",
        per_sample_weights_arg->toString());
  }

  int64_t numIndices = indices.size(0);
  int64_t numBags = offsets.size(0);

  if (include_last_offset) {
    // Check https://github.com/pytorch/pytorch/issues/29019
    // We plan to add one more element in offsets, which is equal to the size of
    // indices. Currently for cuda devices, we still use the legacy
    // implementation even this flag is enabled.
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }
  int64_t featureSize = weight.size(1);
  auto bag_size = at::empty(offsets.sizes(), indices.options());
  auto offset2bag = at::empty(
      {indices.size(0)}, indices.options()); // offset2bag = [0 0 0 0 0]
  auto output = at::empty({numBags, featureSize}, weight.options());

  at::Tensor max_indices;
  if (mode == MODE_MAX) {
    max_indices = at::empty({numBags, featureSize}, indices.options());
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::empty({0}, indices.options());
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "embedding_bag_mlu",
      [&]() {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_mlu", [&]() {
              cnnl_embedding_bag_internal(
                  weight,
                  indices,
                  offsets,
                  per_sample_weights_contiguous,
                  output,
                  offset2bag,
                  bag_size,
                  max_indices,
                  scale_grad_by_freq,
                  mode,
                  sparse,
                  include_last_offset,
                  padding_idx);
            });
      });

  if (indices.numel() == 0) {
    output.fill_(0);
  }
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

at::Tensor cnnl__embedding_bag_dense_backward(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor& maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const std::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward in
  // EmbeddingBag.cpp.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.
  auto grad_contiguous = cnnl_contiguous(grad);
  auto indices_arg = at::TensorArg(indices, "indices", 1);
  auto grad_arg = at::TensorArg(grad, "grad", 1);
  torch_mlu::checkSameMLU("cnnl_embedding_bag_cuda", grad_arg, indices_arg);
  TORCH_CHECK(
      mode == 0,
      "torch_mlu embedding_bag_dense_backward \
                only support CNNL_REDUCEMODE_SUM.");
  TORCH_CHECK(
      scale_grad_by_freq == false,
      "torch_mlu embedding_bag_dense_backward \
                scale_grad_by_freq only support false.");
  TORCH_CHECK(
      padding_idx == -1,
      "torch_mlu embedding_bag_dense_backward \
                unsupport padding_idx.");
  auto output =
      at::zeros({num_weights, grad.size(1)}, grad_contiguous.options());
  int64_t offsets_shape = indices.dim() == 1 ? grad.size(0) : 0;
  auto offsets = at::empty({offsets_shape}, indices.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_backward_mlu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_backward_mlu", [&]() {
              cnnl_embedding_bag_dense_backward_internal(
                  grad_contiguous,
                  indices,
                  offsets,
                  offset2bag,
                  bag_size,
                  per_sample_weights,
                  maximum_indices,
                  output,
                  mode);
            });
      });

  return output;
}

} // namespace ops
} // namespace torch_mlu
