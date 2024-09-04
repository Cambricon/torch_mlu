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

#include <algorithm>
#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

void cnnl_embedding_bag_internal(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& per_sample_weights,
    at::Tensor& output,
    at::Tensor& offset2bag,
    at::Tensor& bag_size,
    at::Tensor& max_indices,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    bool include_last_offset,
    int64_t padding_idx) {
  // prepare cnnl input
  auto weight_impl = getMluTensorImpl(weight);
  auto weight_ptr = mlu_data_ptr(weight_impl);
  CnnlTensorDescriptor weight_desc;
  weight_desc.set(weight, CNNL_LAYOUT_ARRAY);

  auto indices_impl = getMluTensorImpl(indices);
  auto indices_ptr = mlu_data_ptr(indices_impl);
  CnnlTensorDescriptor indices_desc;
  indices_desc.set(indices, CNNL_LAYOUT_ARRAY);

  auto offsets_impl = getMluTensorImpl(offsets);
  auto offsets_ptr = mlu_data_ptr(offsets_impl);
  CnnlTensorDescriptor offsets_desc;
  offsets_desc.set(offsets, CNNL_LAYOUT_ARRAY);

  void* per_sample_weights_ptr = NULL;
  CnnlTensorDescriptor per_sample_weights_desc;
  if (per_sample_weights.defined()) {
    auto per_sample_weights_impl = getMluTensorImpl(per_sample_weights);
    per_sample_weights_ptr = mlu_data_ptr(per_sample_weights_impl);
    per_sample_weights_desc.set(per_sample_weights, CNNL_LAYOUT_ARRAY);
  }
  // prepare cnnl output
  auto size = indices.sizes().vec();
  auto indices_dim = indices.dim();
  if (indices_dim == 0 && indices.numel() == 1) {
    size = std::vector<int64_t>{1};
  }
  for (auto d : weight.sizes().slice(1)) {
    size.push_back(d);
  }

  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  CnnlTensorDescriptor output_desc;
  output_desc.set(output, CNNL_LAYOUT_ARRAY);

  auto offset2bag_impl = getMluTensorImpl(offset2bag);
  auto offset2bag_ptr = mlu_data_ptr(offset2bag_impl);
  CnnlTensorDescriptor offset2bag_desc;
  offset2bag_desc.set(offset2bag, CNNL_LAYOUT_ARRAY);

  auto bag_size_impl = getMluTensorImpl(bag_size);
  auto bag_size_ptr = mlu_data_ptr(bag_size_impl);
  CnnlTensorDescriptor bag_size_desc;
  bag_size_desc.set(bag_size, CNNL_LAYOUT_ARRAY);

  // call cnnl embedding interface
  cnnlReduceMode_t reduce_mode;
  if (mode == MODE_SUM) {
    reduce_mode = CNNL_REDUCEMODE_SUM;
  } else if (mode == MODE_MEAN) {
    reduce_mode = CNNL_REDUCEMODE_MEAN;
  } else {
    reduce_mode = CNNL_REDUCEMODE_MAX;
  }
  CnnlEmbeddingBagDescriptor embeddingBag_desc;
  embeddingBag_desc.set(
      reduce_mode,
      nullptr,
      nullptr,
      nullptr,
      scale_grad_by_freq,
      include_last_offset);

  auto handle = getCurrentHandle();
  // max_indices unsupport
  TORCH_CNNL_CHECK(cnnlEmbeddingBag_v2(
      /* handle                 */ handle,
      /* EmbeddingBag_desc      */ embeddingBag_desc.desc(),
      /* filter_desc            */ weight_desc.desc(),
      /* *filter                */ weight_ptr,
      /* indices_desc           */ indices_desc.desc(),
      /* *indices               */ indices_ptr,
      /* offset_desc            */ offsets_desc.desc(),
      /* *offset                */ offsets_ptr,
      /* per_sample_filter_desc */ per_sample_weights_desc.desc(),
      /* *per_sample_filter     */ per_sample_weights_ptr,
      /* output_desc            */ output_desc.desc(),
      /* *output                */ output_ptr,
      /* offset2bag_desc        */ offset2bag_desc.desc(),
      /* *offset2bag            */ offset2bag_ptr,
      /* bag_size_desc          */ bag_size_desc.desc(),
      /* *bag_size              */ bag_size_ptr,
      /* max_indices_desc       */ NULL,
      /* *max_indices           */ NULL));
  if (indices_dim == 0 && indices.numel() == 1) {
    output.squeeze_(0);
  }
}

void cnnl_embedding_bag_dense_backward_internal(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor& per_sample_weights,
    const at::Tensor& max_indices,
    at::Tensor& output,
    int64_t mode) {
  // prepare cnnl input
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_ptr = mlu_data_ptr(grad_impl);
  CnnlTensorDescriptor grad_desc;
  grad_desc.set(grad, CNNL_LAYOUT_ARRAY);

  auto indices_impl = getMluTensorImpl(indices);
  auto indices_ptr = mlu_data_ptr(indices_impl);
  CnnlTensorDescriptor indices_desc;
  indices_desc.set(indices, CNNL_LAYOUT_ARRAY);

  auto offsets_impl = getMluTensorImpl(offsets);
  auto offsets_ptr = mlu_data_ptr(offsets_impl);
  CnnlTensorDescriptor offsets_desc;
  offsets_desc.set(offsets, CNNL_LAYOUT_ARRAY);

  auto offset2bag_impl = getMluTensorImpl(offset2bag);
  auto offset2bag_ptr = mlu_data_ptr(offset2bag_impl);
  CnnlTensorDescriptor offset2bag_desc;
  offset2bag_desc.set(offset2bag, CNNL_LAYOUT_ARRAY);

  auto bag_size_impl = getMluTensorImpl(bag_size);
  auto bag_size_ptr = mlu_data_ptr(bag_size_impl);
  CnnlTensorDescriptor bag_size_desc;
  bag_size_desc.set(bag_size, CNNL_LAYOUT_ARRAY);

  void* per_sample_weights_ptr = NULL;
  CnnlTensorDescriptor per_sample_weights_desc;
  if (per_sample_weights.defined()) {
    auto per_sample_weights_impl = getMluTensorImpl(per_sample_weights);
    per_sample_weights_ptr = mlu_data_ptr(per_sample_weights_impl);
    per_sample_weights_desc.set(per_sample_weights, CNNL_LAYOUT_ARRAY);
  }

  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  CnnlTensorDescriptor output_desc;
  output_desc.set(output, CNNL_LAYOUT_ARRAY);

  cnnlReduceMode_t reduce_mode;
  if (mode == 0) {
    reduce_mode = CNNL_REDUCEMODE_SUM;
  } else if (mode == 1) {
    reduce_mode = CNNL_REDUCEMODE_MEAN;
  } else {
    reduce_mode = CNNL_REDUCEMODE_MAX;
  }
  CnnlEmbeddingBagDescriptor embeddingBag_desc;
  embeddingBag_desc.set(reduce_mode, nullptr, nullptr, nullptr, false, false);

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlEmbeddingBagBackward(
      /* handle                 */ handle,
      /* embeddingBag_desc      */ embeddingBag_desc.desc(),
      /* diff_desc              */ grad_desc.desc(),
      /* *diff                  */ grad_ptr,
      /* indices_desc           */ indices_desc.desc(),
      /* *indices               */ indices_ptr,
      /* offset_desc            */ indices.dim() == 1 ? offsets_desc.desc()
                                                      : NULL,
      /* *offset                */ indices.dim() == 1 ? offsets_ptr : NULL,
      /* offset2bag_desc        */ offset2bag_desc.desc(),
      /* *offset2bag            */ offset2bag_ptr,
      /* bag_size_desc          */ bag_size_desc.desc(),
      /* *bag_size              */ bag_size_ptr,
      /* per_sample_filter_desc */ per_sample_weights_desc.desc(),
      /* *per_sample_filter     */ per_sample_weights_ptr,
      /* max_indices_desc       */ NULL,
      /* *max_indices           */ NULL,
      /* output_desc            */ output_desc.desc(),
      /* *output                */ output_ptr));
}

} // namespace ops
} // namespace torch_mlu
