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
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

void cnnl_scatter_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    cnnlScatterMode_t mode) {
  auto self_impl = getMluTensorImpl(self);
  auto self_desc = getTensorDesc(self_impl, CNNL_LAYOUT_ARRAY);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto index_impl = getMluTensorImpl(index);
  auto index_desc = getTensorDesc(index_impl, CNNL_LAYOUT_ARRAY);
  auto index_ptr = mlu_data_ptr(index_impl);

  auto src_impl = getMluTensorImpl(src);
  auto src_desc = getTensorDesc(src_impl, CNNL_LAYOUT_ARRAY);
  auto src_ptr = mlu_data_ptr(src_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  size_t space_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetScatterWorkspaceSize(handle, index_desc.get(), &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
  if (mode == CNNL_SCATTER) {
    // complex, bf16 are not supported
    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        self.scalar_type(),
        "MLU scatter",
        [&] {
          TORCH_CNNL_CHECK(cnnlScatter_v2(
              handle,
              dim,
              self_desc.get(),
              self_ptr,
              index_desc.get(),
              index_ptr,
              src_desc.get(),
              src_ptr,
              workspace_ptr.get(),
              space_size,
              output_desc.get(),
              output_ptr,
              mode));
        });
  }
  if (mode == CNNL_SCATTER_ADD || mode == CNNL_SCATTER_MAX ||
      mode == CNNL_SCATTER_MIN) {
    // complex, int8, uint8, int16, bool are not supported
    AT_DISPATCH_MLU_FLOAT_HALF_INT_AND_BFLOAT16(
        self.scalar_type(), "MLU scatter_reduce", [&] {
          TORCH_CNNL_CHECK(cnnlScatter_v2(
              handle,
              dim,
              self_desc.get(),
              self_ptr,
              index_desc.get(),
              index_ptr,
              src_desc.get(),
              src_ptr,
              workspace_ptr.get(),
              space_size,
              output_desc.get(),
              output_ptr,
              mode));
        });
  }
}

} // namespace ops
} // namespace torch_mlu
