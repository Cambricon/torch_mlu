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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_histc_internal(
    const at::Tensor self,
    int64_t bins,
    float minvalue,
    float maxvalue,
    const at::Tensor output) {
  // create cnnlHistogramdescriptor
  cnnlHistogramMode_t mode = CNNL_HISTOGRAM_MODE_HISTO_COUNT;
  CnnlHistogramDescriptor histc_desc;
  histc_desc.set(mode, bins, minvalue, maxvalue, false, false);

  // worksize calculate
  auto handle = getCurrentHandle();

  CnnlTensorDescriptor self_desc;
  // only need dummy desc becasause cnnl api requires it
  CnnlTensorDescriptor dummy_desc;
  CnnlTensorDescriptor output_desc;
  self_desc.set(self);
  output_desc.set(output);

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetHistogramWorkspaceSize(
      handle, self_desc.desc(), dummy_desc.desc(), &workspace_size));
  auto self_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  auto self_ptr = mlu_data_ptr(self_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // invoke kernal cnnlHistc
  TORCH_CNNL_CHECK(cnnlHistc(
      handle,
      histc_desc.desc(),
      self_desc.desc(),
      self_ptr,
      workspace_ptr.get(),
      workspace_size,
      output_desc.desc(),
      output_ptr));
}
} // namespace ops
} // namespace torch_mlu
