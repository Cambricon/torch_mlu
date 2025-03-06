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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymIntArrayRef.h"
#include "ATen/SparseCsrTensorUtils.h"

namespace torch_mlu {
namespace ops {

// slice_backward_without_cnnl is aligned with the native implementation.
// Compared with the native implementation, slice_backward_without_cnnl reduces
// the dispatch process of the host-side operator to improve performance.
at::Tensor slice_backward_without_cnnl(
    const at::Tensor& grad_output,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step,
    at::Tensor& grad_input) {
  auto res_input = grad_input.slice(dim, start, end, step);
  torch_mlu::ops::cnnl_copy_(res_input, grad_output);
  return grad_input;
}

// cnnl_slice_backward uses cnnlStridedSliceBackward to perform backward
// calculations. Compared with slice_backward_without_cnnl, it reduces the two
// invokes (fill-op and copy-op) to one invoke (slice_backward op) to improve
// performance.
at::Tensor cnnl_slice_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef input_sizes,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  auto grad_options = grad_output.options();
  std::optional<at::ScalarType> dtype_opt =
      c10::optTypeMetaToScalarType(grad_options.dtype_opt());
  std::optional<at::Layout> layout_opt = grad_options.layout_opt();
  std::optional<at::Device> device_opt = grad_options.device_opt();
  std::optional<bool> pin_memory_opt = grad_options.pinned_memory_opt();
  at::Layout layout_ = layout_opt.value_or(at::Layout::Strided);
  if (at::sparse_csr::is_sparse_compressed(layout_)) {
    auto grad_input = at::native::zeros_symint(
        c10::fromIntArrayRefSlow(input_sizes),
        dtype_opt,
        layout_opt,
        device_opt,
        pin_memory_opt);
    return slice_backward_without_cnnl(
        grad_output, dim, start, end, step, grad_input);
  }

  auto grad_input = torch_mlu::ops::cnnl_empty(
      input_sizes, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // cnnl does not support tensors larger than 2GB.
  constexpr int64_t TWO_GB = 2LL * 1024 * 1024 * 1024;
  if ((grad_output.nbytes() >= TWO_GB || grad_input.nbytes() >= TWO_GB) ||
      at::isComplexType(grad_output.scalar_type())) {
    torch_mlu::ops::cnnl_zero_(grad_input);
    return slice_backward_without_cnnl(
        grad_output, dim, start, end, step, grad_input);
  }

  if (grad_output.numel() == 0) {
    return cnnl_zero_(grad_input);
  }

  dim = at::maybe_wrap_dim(dim, grad_input.dim());
  auto ndim = input_sizes.size();
  TORCH_CHECK_INDEX(ndim != 0, "slice() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK(step > 0, "slice step must be positive");

  auto grad_output_contiguous = cnnl_contiguous(grad_output);
  auto grad_input_contiguous = cnnl_contiguous(grad_input);

  // Slice tensor like t[:], mean start=0, end = INT64_MAX.
  // Since cnnlStridedSliceBackward requires the input type to be int,
  // end=INT64_MAX needs to be processed separately at this time.
  end = (end >= input_sizes[dim]) ? input_sizes[dim] : end;

  std::vector<int> begins(ndim, 0);
  std::vector<int> ends(ndim, 0);
  std::vector<int> strides(ndim, 1);

  // if n != dim, then ends[n] = input_sizes[n]
  for (int i = 0; i < ndim; ++i) {
    ends[i] = input_sizes[i];
  }

  begins[dim] = start;
  ends[dim] = end;
  strides[dim] = step;

  cnnl_slice_backward_internal(
      grad_output_contiguous, begins, ends, strides, grad_input_contiguous);

  if (is_copy_necessary(grad_input, grad_input_contiguous)) {
    grad_input.copy_(grad_input_contiguous);
  }

  return grad_input;
};

} // namespace ops
} // namespace torch_mlu