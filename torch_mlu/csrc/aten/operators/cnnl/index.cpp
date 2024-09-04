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

#include "ATen/native/IndexingUtils.h"
#include "ATen/MemoryOverlap.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/index_utils.h"

namespace torch_mlu {
namespace ops {

// Type list supported by GPU.
// coda path: aten/src/ATen/native/cuda/IndexKernel.cu
// index type list:        at::ScalarType::Byte, at::ScalarType::Long,
//                         at::ScalarType::Bool
// input/output type list: at::ScalarType::Byte, at::ScalarType::Char,
//                         at::ScalarType::Int, at::ScalarType::Long,
//                         at::ScalarType::Short, at::ScalarType::Double,
//                         at::ScalarType::Float, at::ScalarType::ComplexDouble,
//                         at::ScalarType::ComplexFloat,
//                         at::ScalarType::ComplexHalf, at::ScalarType::Half,
//                         at::ScalarType::Bool, at::ScalarType::BFloat16.

// Type list supported by CNNL index select.
// index type list:        at::ScalarType::Byte, at::ScalarType::Long,
//                         at::ScalarType::Bool
// input/output type list: at::ScalarType::Byte, at::ScalarType::Char,
//                         at::ScalarType::Short, at::ScalarType::Int,
//                         at::ScalarType::Half, at::ScalarType::Bool,
//                         at::ScalarType::Float, at::ScalarType::Long
//                         at::ScalarType::Double
// Not real support double, but support double to float convert.

// When indices are bool: the dim size of each indice equals to the
// corresponding dim size of input，the number of indices is less equal to input
// There are several modes supported by AdvancedIndex for bool indices:
//
// 1. a[b, c]，eg: a.shape = [m, m] b.shape = [m] c.shape=[m], the output.
// shape maximum is [m]。This situation is special，because the shapes of b and
// c must be mutually broadcastable.
// 2. a[b], eg: a.shape = [m, n] b.shape = [m, n] output.shape maximum is [m *
// n], or a.shape = [m, n], b.shape = [m], output.shape is [m, n]
// 3. a[b,:], eg: a.shape = [m, n], b.shape = [m], output.shape maximum is [m,
// n] (same as mode 2)
//
// When indices are long: each value of every indices
// must be less equal to the maximum of input dims
// and support negative number (The absolute number of indices value should also
// less equal!) There are several modes supported by AdvancedIndex for long
// indices:
//
// 1. a[b,c], eg. a.shape = [n,m], b.shape = [n',m'], c.shape = [n',m'],
// output.shape = [n',m']
// 2. a[b], eg. a.shape = [n,m,q], b.shape = [n',m'], output.shape = [n',m',m,q]
// 3. a[b,:], eg. a.shape = [n,m,q], b.shape = [n',m'], output.shape =
// [n',m',m,q] (same as mode 2)
// 4. a[b,:,c], this mode equals to a[b,c] plus a transpotation to input tensor:
// eg. a.shape = [n,m,q], b.shape = [n',q'], c.shape=[n',q'], output.shape =
// [n',q',m]

// compute the output shape and broadcast shape of Bool indices
std::vector<int64_t> compute_shapes(
    const at::Tensor& self,
    const at::Tensor& indice) {
  std::vector<int64_t> output_dims;
  auto self_size = self.sizes().vec();
  output_dims.emplace_back(indice.numel());
  if (indice.dim() != self.dim()) {
    for (int64_t i = indice.dim(); i < self.dim(); ++i) {
      output_dims.emplace_back(self_size[i]);
    }
  }
  return output_dims;
}

static std::string shapes_as_str(at::TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

// This function is used to restride the source Tensor. This function creates a
// view of part of the source tensor so that the tensor have the same shape with
// the value tensor. For example, suppose we have a tensor self =
// [[1,2,3,4],[5,6,7,8],[9,10,11,12]], value = [[0,0,0],[0,0,0]] and indices =
// [0,1], the result of restride_src is [[1,2,3],[1,2,3]]. In the native
// PyTorch, the restrided tensor is used as input to kernels, however, we only
// use this the restrided tensor to do parameter check.
static at::Tensor restride_src(
    const at::Tensor& src,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = at::DimVector(src.sizes());
  auto strides = at::DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(
      shape.begin() + dims_before,
      replacement_shape.begin(),
      replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

static at::Tensor reshape_indexer(
    const at::Tensor& index,
    int64_t dims_before,
    int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = at::DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

MLUAdvancedIndex::MLUAdvancedIndex(
    const at::Tensor& src,
    at::TensorList indices_list,
    bool hasContiguousSubspace,
    std::vector<bool> hasDefined) {
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (size_t dim = 0; dim < indices_list.size(); dim++) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) !=
          indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
          replacement_shape.end()) {
    TORCH_CHECK_INDEX(
        false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);
  this->self = src;
  this->hasContiguousSubspace = hasContiguousSubspace;
  this->hasDefined = hasDefined;
  for (auto& index : indices_list) {
    indices.push_back(index);
  }
}

// Here we use a MLUAdvancedIndex structure rather than a tuple which consist of
// self tensor and indices tensor because we need paramaters like dims before
// and dims after to generate info.src which is used to perform paramater check.
MLUAdvancedIndex make_info(
    at::Tensor self,
    const c10::List<std::optional<at::Tensor>>& orig) {
  checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = at::native::expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = at::expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(
        false,
        "shape mismatch: indexing tensors could not be broadcast together"
        " with shapes ",
        shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  std::vector<bool> has_defined;
  for (at::Tensor idx : indices) {
    if (idx.defined()) {
      has_defined.emplace_back(true);
    } else {
      has_defined.emplace_back(false);
    }
  }
  // transpose self and indices together so that they're adjacent at the front
  bool hasContiguousSubspace = true;
  if (!at::native::hasContiguousSubspace(indices)) {
    hasContiguousSubspace = false;
    std::tie(self, indices) = at::native::transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined() && indices[i].device() != self.device()) {
      indices[i] = indices[i].to(self.device());
    }
  }
  return MLUAdvancedIndex(self, indices, hasContiguousSubspace, has_defined);
}

// Almost copy from aten/src/ATen/native/TensorAdvancedIndexing.cpp.
// Modify indices type for reduce call materialize() function.
void check_indices_on_cpu_or_selfdevice(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  auto dev = self.device();
  bool indices_on_cpu_or_dev = std::all_of(
      indices.begin(),
      indices.end(),
      [=](const std::optional<at::Tensor>& opt) {
        return opt.has_value() && opt->defined()
            ? (opt->is_cpu() || opt->device() == dev)
            : true;
      });
  TORCH_CHECK(
      indices_on_cpu_or_dev,
      "indices should be either on ",
      at::kCPU,
      " or on the same device as the indexed tensor (",
      dev,
      ")");
}

// MLU index op most functions is implemented by CNNL kernel. This is
// very different with pytorch GPU/CPU index op. In pytorch side, index
// op most functions is completed in CPU code, device kernel only handle
// long index tensor and using offset to accomplish kernel function.
// Code path: aten/src/ATen/native/TensorAdvancedIndexingUtils.h

// There are several situation of indices.
// 1) single index tensor in indices, CNNL kernel will do the same
//    things as pytorch;
// 2) several index tensors in indices:
// 2.1) All index tensors type is long, CNNL will do the same
//    things as pytorch;
// 2.2) Otherwise TORCH_MLU need to almost fellow pytorch make_info function,
//      and then call CNNL kernel.
// For 2.2, CNNL will support later.
at::Tensor& cnnl_index_out(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    at::Tensor& output) {
  auto dtype = self.scalar_type();
  TORCH_CHECK(
      (at::isIntegralType(dtype, /*includeBool=*/true) ||
       at::isFloatingType(dtype)),
      "Torch mlu only support integer type, floating type",
      "and bool type, but now get dtype: ",
      dtype);
  static constexpr c10::string_view DIM_WARNING =
      "Tensor too large or too many (> 8) dimensions";
  TORCH_CHECK(self.dim() <= CNNL_MAX_DIM_SIZE, DIM_WARNING);

  TORCH_CHECK_INDEX(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  // Only allow: `dev_tensor[{cpu,dev}_tensor]`.
  // See: https://github.com/pytorch/pytorch/pull/69607
  check_indices_on_cpu_or_selfdevice(self, indices);

  if (output.defined()) {
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, self);
    for (int i = 0; i < indices.size(); ++i) {
      if (indices[i].has_value()) {
        at::assert_no_overlap(output, indices[i].value());
      }
    }
  }

  bool is_include_bool = std::any_of(
      indices.begin(),
      indices.end(),
      [](const std::optional<at::Tensor>& opt_tensor) {
        if (opt_tensor.has_value() &&
            (opt_tensor->scalar_type() == at::kByte ||
             opt_tensor->scalar_type() == at::kBool))
          return true;
        return false;
      });
  // TODO(CNNLCORE-13368): Using pytorch make info to convert bool(byte) index
  // tensor to long index tensor. Kernel is not support now, for
  // multi-bool(byte) index tensors or mix index tensors with long index tensor
  // and bool(byte) index tensor.
  if (is_include_bool) {
    auto info = make_info(self, indices);
    auto self_contiguous = cnnl_contiguous(info.self);
    const int index_size = info.indices.size();
    std::vector<at::Tensor> indices_expand(index_size, at::Tensor());
    for (int i = 0; i < index_size; ++i) {
      indices_expand[i] = std::move(cnnl_contiguous(info.indices[i]));
    }
    return cnnl_index_internal(output, self_contiguous, indices_expand);
  } else {
    auto self_contiguous = cnnl_contiguous(self);
    const int index_size = indices.size();
    std::vector<at::Tensor> indices_expand(index_size, at::Tensor());
    auto self_device = self.device();
    for (int i = 0; i < index_size; ++i) {
      if (indices[i].has_value() && indices[i].value().defined()) {
        // Ensure indices are on the same device as self
        indices_expand[i] = cnnl_contiguous(
            indices[i].value().device() != self_device
                ? indices[i].value().to(self_device)
                : indices[i].value());
      }
    }
    return cnnl_index_internal(output, self_contiguous, indices_expand);
  }
}

at::Tensor cnnl_index(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  at::Tensor output = at::empty({0}, self.options());
  output = cnnl_index_out(self, indices, output);
  return output;
}

} // namespace ops
} // namespace torch_mlu
