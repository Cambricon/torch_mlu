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

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_polar_internal(
    const at::Tensor& abs,
    const at::Tensor& angle,
    at::Tensor& output) {
  // [in] handle
  auto handle = getCurrentHandle();

  // [in] abs_desc & [in] abs
  auto abs_impl = getMluTensorImpl(abs);
  auto abs_desc = getTensorDesc(abs_impl, CNNL_LAYOUT_ARRAY);
  auto abs_ptr = mlu_data_ptr(abs_impl);

  // [in] angle_desc & [in] angle
  auto angle_impl = getMluTensorImpl(angle);
  auto angle_desc = getTensorDesc(angle_impl, CNNL_LAYOUT_ARRAY);
  auto angle_ptr = mlu_data_ptr(angle_impl);

  // [in] output_desc & [out] output
  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // [in] workspace & [in] workspace_size
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetPolarWorkspaceSize(
      handle,
      abs_desc.get(),
      angle_desc.get(),
      output_desc.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // polar operation
  TORCH_CNNL_CHECK(cnnlPolar(
      handle,
      abs_desc.get(),
      abs_ptr,
      angle_desc.get(),
      angle_ptr,
      output_desc.get(),
      output_ptr,
      workspace_ptr.get(),
      workspace_size));

  return output;
}

} // namespace ops
} // namespace torch_mlu
