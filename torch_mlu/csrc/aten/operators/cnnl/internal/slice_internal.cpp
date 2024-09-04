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

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <c10/core/Storage.h>
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

inline static void call_slice_without_any_check(
    at::Tensor& output,
    const at::Tensor& input,
    const int64_t* start,
    const int64_t* end,
    const int64_t* step,
    c10::MemoryFormat memory_format) {
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto layout = suggestCnnlLayout(memory_format);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlStridedSlice_v2(
      handle,
      input_desc.get(),
      input_ptr,
      start,
      end,
      step,
      output_desc.get(),
      output_ptr));
}

// slice internal parameter need be legal, and parameters are processed in
// cnnl_slice.
at::Tensor cnnl_slice_internal(
    const at::Tensor& input,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  auto ndim = input.dim();
  TORCH_MLU_CHECK(ndim > 0, "slice() cannot be applied to a 0-dim tensor.");
  TORCH_MLU_CHECK(step > 0, "slice step must be positive");
  // The following 3 param check are required by cnnl kernels
  TORCH_MLU_CHECK(
      dim >= 0 && dim < ndim,
      "slice dim must be non ",
      "negative and the value of dim must less than the dims of input");
  auto sizes = input.sizes().vec();
  TORCH_MLU_CHECK(
      start >= 0 && end >= start && end <= sizes[dim],
      "when using slice, values of start and end need to meet the following ",
      "conditions: 0 <= start <= end <=sizes[dim]");
  auto memory_format = input.suggest_memory_format();
  TORCH_MLU_CHECK(
      input.is_contiguous(memory_format),
      "Input tensor needs ",
      "to be contiguous.");

  // currently we support partial storage sharing
  sizes[dim] = (end - start + step - 1) / step;
  if (dim == 0 && step == 1) {
    auto inplace_output = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(input.storage()), input.key_set(), input.dtype());
    at::native::setStrided(
        inplace_output,
        at::IntArrayRef(sizes),
        input.strides(),
        input.storage_offset() + start * input.stride(0));
    return inplace_output;
  }

  if (step == 1 && end - start == input.size(dim))
    return input;

  at::Tensor output = at::empty(sizes, input.options(), memory_format);
  if (output.numel() == 0) {
    return output;
  }

  // Modify size info for create desc.
  dim = modify_dim_based_on_layout(dim, memory_format);
  const auto& input_size =
      modify_dims_based_on_layout(input.sizes(), memory_format);

  // Modify cnnl start/end/step info based on real layout.
  std::vector<int64_t> starts(ndim, 0);
  std::vector<int64_t> ends(input_size);
  std::vector<int64_t> steps(ndim, 1);
  starts[dim] = start;
  ends[dim] = end;
  steps[dim] = step;

  call_slice_without_any_check(
      output, input, starts.data(), ends.data(), steps.data(), memory_format);
  return output;
}

// cnnl_slice_internal_v2 is for multi-dims slice.
// vector<int> dims is slicing dim collect. a[1:5:2, 3:20:1, ...] will be stored
// like dims == {1, 2}; start == {1, 3}; end == {5, 10}; step == {2, 1}.
// (TODO)shangang: not support replaceSliceToReshape now, so don't support out
// interface now.
at::Tensor cnnl_multi_dims_slice_internal(
    const at::Tensor& input,
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps) {
  const int ndim = input.dim();
  const int slice_dim = dims.size();
  TORCH_MLU_CHECK(
      slice_dim > 0 && slice_dim <= ndim,
      "When using multi dims slice, size of dims needs to be greater than zero, ",
      "and less or equal to dims of input.");
  TORCH_MLU_CHECK(
      slice_dim == starts.size(),
      "The size of starts needs to be ",
      "equal to the size with dims.");
  TORCH_MLU_CHECK(
      slice_dim == ends.size(),
      "The size of ends needs to be equal ",
      "to the size with dims.");
  TORCH_MLU_CHECK(
      slice_dim == steps.size(),
      "The size of steps needs to be ",
      "equal to the size with dims.");

  auto memory_format = input.suggest_memory_format();
  TORCH_MLU_CHECK(
      input.is_contiguous(memory_format),
      "Input tensor needs ",
      "to be contiguous.");

  auto output_size = input.sizes().vec();
  // Shared storage.
  {
    auto caculate_output_size =
        [&output_size](
            int64_t dim, int64_t start, int64_t end, int64_t step) -> void {
      output_size[dim] = (end - start + step - 1) / step;
    };

    // Just slice first dim and step is equal to 1. Using set a new offset to
    // shared storage.
    if (slice_dim == 1 && dims[0] == 0 && steps[0] == 1) {
      caculate_output_size(dims[0], starts[0], ends[0], steps[0]);
      auto inplace_output = at::detail::make_tensor<c10::TensorImpl>(
          c10::Storage(input.storage()), input.key_set(), input.dtype());
      at::native::setStrided(
          inplace_output,
          at::IntArrayRef(output_size),
          input.strides(),
          input.storage_offset() + starts[0] * input.stride(0));
      return inplace_output;
    }

    // output size is equal with input size, just return input.
    for (int i = 0; i < dims.size(); ++i) {
      caculate_output_size(dims[i], starts[i], ends[i], steps[i]);
    }
    if (input.sizes().vec() == output_size)
      return input;
  }

  // Create output tensor
  at::Tensor output = at::empty(output_size, input.options(), memory_format);
  // 0-element check
  if (output.numel() == 0) {
    return output;
  }

  // Modify size info for create desc.
  auto input_size_based_on_mf =
      modify_dims_based_on_layout(input.sizes(), memory_format);

  // Modify cnnl start/end/step info based on real layout.
  std::vector<int64_t> cnnl_start(ndim, 0);
  std::vector<int64_t> cnnl_end(input_size_based_on_mf);
  std::vector<int64_t> cnnl_step(ndim, 1);

  // Modify dims based on memory format;
  for (int i = 0; i < dims.size(); ++i) {
    int64_t dim = modify_dim_based_on_layout(dims[i], memory_format);
    cnnl_start[dim] = starts[i];
    cnnl_end[dim] = ends[i];
    cnnl_step[dim] = steps[i];
  }

  call_slice_without_any_check(
      output,
      input,
      cnnl_start.data(),
      cnnl_end.data(),
      cnnl_step.data(),
      memory_format);
  return output;
}

} // namespace ops
} // namespace torch_mlu
