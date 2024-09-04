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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {
void cnnl_index_copy_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source) {
  dim = at::maybe_wrap_dim(dim, input);

  // set descriptor config
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  CnnlTensorDescriptor output_desc;
  output_desc.set(output);

  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = mlu_data_ptr(input_impl);
  CnnlTensorDescriptor input_desc;
  input_desc.set(input);

  auto long_index = at::empty_like(index);
  auto index_impl = getMluTensorImpl(long_index);
  setCnnlType(index_impl, CNNL_DTYPE_INT64);
  auto index_ptr = mlu_data_ptr(index_impl);
  cnnl_cast_internal(index, long_index);
  CnnlTensorDescriptor index_desc;
  index_desc.set(long_index);

  auto source_impl = getMluTensorImpl(source);
  auto source_ptr = mlu_data_ptr(source_impl);
  CnnlTensorDescriptor source_desc;
  source_desc.set(source);

  // prepare workspace
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetIndexCopyWorkspaceSize(
      handle, index_desc.desc(), &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  bool deterministic_mode = true;
  TORCH_CNNL_CHECK(cnnlIndexCopy(
      handle,
      static_cast<int32_t>(dim),
      deterministic_mode,
      input_desc.desc(),
      input_ptr,
      source_desc.desc(),
      source_ptr,
      index_desc.desc(),
      index_ptr,
      workspace_ptr.get(),
      workspace_size,
      output_desc.desc(),
      output_ptr));
}

} // namespace ops
} // namespace torch_mlu
