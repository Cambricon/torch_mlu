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

#include "aten/utils/internal_util.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/utils/binaryops_util.h"

// TODO(sfengyang): the mapping tool support NCL cnnl layout in the future.
static std::map<cnnlTensorLayout_t, std::vector<int64_t>> layout2index = {
    {CNNL_LAYOUT_NCHW, {0, 1, 2, 3}},
    {CNNL_LAYOUT_NHWC, {0, 2, 3, 1}},
    {CNNL_LAYOUT_HWCN, {2, 3, 1, 0}},
    {CNNL_LAYOUT_NCDHW, {0, 1, 2, 3, 4}},
    {CNNL_LAYOUT_NDHWC, {0, 2, 3, 4, 1}},
    {CNNL_LAYOUT_NTC, {0, 1, 2}},
    {CNNL_LAYOUT_TNC, {1, 0, 2}}};

namespace torch_mlu {

void transLayoutParameterDim(
    const cnnlTensorLayout_t& source_laytout,
    const cnnlTensorLayout_t& target_layout,
    const int64_t& in_dim,
    int64_t* out_dim) {
  std::vector<int64_t> source_dims_index_vec_;
  std::vector<int64_t> target_dims_index_vec_;
  auto source_search = layout2index.find(source_laytout);
  auto target_search = layout2index.find(target_layout);
  int64_t non_negative_dim = in_dim;
  if (source_search != layout2index.end()) {
    source_dims_index_vec_ = source_search->second;
  } else {
    TORCH_CHECK(
        false,
        "source_layout is wrong, source_layout must come from "
        "CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_HWCN, "
        "CNNL_LAYOUT_NDHWC, CNNL_LAYOUT_NCDHW, CNNL_LAYOUT_TNC, CNNL_LAYOUT_NTC");
  }
  if (target_search != layout2index.end()) {
    target_dims_index_vec_ = target_search->second;
  } else {
    TORCH_CHECK(
        false,
        "target_layout is wrong. target_layout must come from "
        "CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_HWCN, "
        "CNNL_LAYOUT_NDHWC,CNNL_LAYOUT_NCDHW, CNNL_LAYOUT_TNC, CNNL_LAYOUT_NTC");
  }
  TORCH_CHECK(
      source_dims_index_vec_.size() == target_dims_index_vec_.size(),
      "source_layout doesn't match target_layout.");
  int64_t max_dim = source_dims_index_vec_.size();
  // remove negative dim
  if (in_dim < 0)
    non_negative_dim = in_dim + max_dim;
  TORCH_CHECK(
      non_negative_dim < max_dim,
      "max_dim need to larger than non_negative_dim.");
  int64_t source_index = source_dims_index_vec_[non_negative_dim];
  auto target_iter = std::find(
      target_dims_index_vec_.begin(),
      target_dims_index_vec_.end(),
      source_index);
  if (target_iter != target_dims_index_vec_.end()) {
    *out_dim = std::distance(target_dims_index_vec_.begin(), target_iter);
  } else {
    TORCH_CHECK(false, "dims trans error!");
  }
}

std::vector<int64_t> modify_dims_based_on_layout(
    const at::IntArrayRef& dim,
    const c10::MemoryFormat memory_format) {
  // dimension is 0, return.
  // dimension == 1 and numel == 1, return.
  if (!dim.size() || (dim.size() == 1 && dim[0] == 1)) {
    return dim.vec();
  }
  std::vector<int64_t> target_dim;
  static std::vector<int> cl_dim_order{0, 2, 3, 1};
  static std::vector<int> cl3d_dim_order{0, 2, 3, 4, 1};
  // trans tensor/stride size to cnnl desc size/stride.
  auto modify_dims_pos = [](const std::vector<int>& dim_order,
                            const at::IntArrayRef& input,
                            std::vector<int64_t>& out) {
    for (const auto& item : dim_order) {
      out.push_back(input[item]);
    }
  };
  switch (memory_format) {
    case c10::MemoryFormat::ChannelsLast:
      TORCH_CHECK(
          dim.size() == 4,
          "dim size must be 4 when memory_format ",
          "is ChannelsLast.");
      modify_dims_pos(cl_dim_order, dim, target_dim);
      break;
    case c10::MemoryFormat::ChannelsLast3d:
      TORCH_CHECK(
          dim.size() == 5,
          "dim size must be 5 when memory_format is ",
          "ChannelsLast3d.");
      modify_dims_pos(cl3d_dim_order, dim, target_dim);
      break;
    case c10::MemoryFormat::Contiguous:
      target_dim = std::move(dim.vec());
      break;
    default:
      TORCH_CHECK(false, "memory format not support.");
      break;
  }
  return target_dim;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_tensor_size_stride(
    const at::Tensor& self,
    at::MemoryFormat memory_format) {
  auto self_size = modify_dims_based_on_layout(self.sizes(), memory_format);
  auto self_stride = get_contiguous_strides(self_size);
  return std::make_tuple(self_size, self_stride);
}

// for now, this function is used to decide whether output and output_contiguous
// are the same
bool is_copy_necessary(
    const at::Tensor& output,
    const at::Tensor& output_contiguous) {
  if (!output.defined() && !output_contiguous.defined())
    return false;
  TORCH_CHECK(
      output.defined() && output_contiguous.defined(),
      "One of those two tensor is undefined.");
  TORCH_CHECK(
      output.sizes() == output_contiguous.sizes(),
      "sizes of two input tensors are not the same.");

  // check if underlying data and strides of these tensors are the same.
  if (output.data_ptr() != output_contiguous.data_ptr() ||
      output.strides() != output_contiguous.strides()) {
    return true;
  }

  // check if dtype are the same.
  if (output.options().dtype() != output_contiguous.options().dtype()) {
    return true;
  }

  return false;
}

// used for structured kernel that out has been computed in meta func.
// output_contiguous = cnnl_contiguous(output) may be not efficient
// while empty is more efficient. In kernel file , copy_ is needed
// when a new empty tensor is created to maintain origin strides.
at::Tensor maybe_create_out(
    const at::Tensor& out,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  if (out.strides() != strides) {
    return at::empty_strided(sizes, strides, options);
  }
  return out;
}

} // namespace torch_mlu
