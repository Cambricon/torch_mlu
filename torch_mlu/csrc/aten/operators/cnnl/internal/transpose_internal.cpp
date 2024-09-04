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

#include <c10/core/Storage.h>
#include <c10/util/Optional.h>

// #include <TH/THTensor.hpp>

#include "ATen/InferSize.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_transpose_internal(
    const at::Tensor& input,
    int64_t dim0,
    int64_t dim1) {
  if (dim0 == dim1) {
    return input;
  }
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);

  const auto input_size = input.sizes();
  const auto dim = input.dim();
  dim0 = ::at::maybe_wrap_dim(dim0, dim);
  dim1 = ::at::maybe_wrap_dim(dim1, dim);
  c10::SmallVector<int64_t, 8> output_size(dim);
  c10::SmallVector<int, 8> order(dim);
  for (auto i = 0; i < dim; ++i) {
    output_size[i] = input_size[i];
    order[i] = i;
  }
  output_size[dim0] = input_size[dim1];
  output_size[dim1] = input_size[dim0];
  order[dim0] = dim1;
  order[dim1] = dim0;
  auto output = at::empty(output_size, input.options());
  if (output.numel() == 0) {
    return output;
  }
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  CnnlTransposeDescriptor trans_desc;
  trans_desc.set(input.dim(), order.data());

  auto handle = getCurrentHandle();
  // Get workspace
  size_t workspace_size = 0;
  cnnlGetTransposeWorkspaceSize(
      handle, input_desc.get(), trans_desc.desc(), &workspace_size);
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlTranspose_v2(
      handle,
      trans_desc.desc(),
      input_desc.get(),
      input_ptr,
      output_desc.get(),
      output_ptr,
      workspace_ptr.get(),
      workspace_size));
  return output;
}

} // namespace ops
} // namespace torch_mlu
