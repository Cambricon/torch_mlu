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
#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_set_(at::Tensor& self, c10::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& cnnl_set_(at::Tensor& result, const at::Tensor& source) {
  return at::native::set_tensor_(result, source);
}

at::Tensor& cnnl_set_(
    at::Tensor& self,
    c10::Storage source,
    int64_t storage_offset,
    at::IntArrayRef size,
    at::IntArrayRef stride) {
  at::native::checkSetStorage(self, source, storage_offset, size, stride);
  getMluTensorImpl(self)->set_storage_offset(storage_offset);
  c10::optional<at::IntArrayRef> stride_opt = stride.data() != nullptr
      ? c10::optional<at::IntArrayRef>(stride)
      : c10::nullopt;
  resize_impl_mlu_(getMluTensorImpl(self), size, stride_opt);
  return self;
}

at::Tensor& cnnl_set_(at::Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  c10::Storage storage(
      c10::Storage::use_byte_size_t(),
      0,
      torch_mlu::MLUCachingAllocator::get(),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

bool cnnl_is_set_to(const at::Tensor& self, const at::Tensor& tensor) {
  return at::native::is_set_to(self, tensor);
}

} // namespace ops
} // namespace torch_mlu
