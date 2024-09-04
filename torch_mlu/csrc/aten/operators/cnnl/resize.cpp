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

#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

void resize_bytes_mlu(c10::StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(
      storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(
      allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();
  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, at::Device(device)));
    storage->set_nbytes(0);
    return;
  }

  const torch_mlu::mlu::MLUGuard device_guard(device);
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr() && storage->nbytes()) {
    // at::globalContext().lazyInitCUDA();
    auto stream = getCurrentMLUStream();
    TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
        data.get(),
        const_cast<void*>(storage->data()),
        std::min(storage->nbytes(), size_bytes),
        stream.stream(),
        CNRT_MEM_TRANS_DIR_DEV2DEV));
  }
  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

const at::Tensor& cnnl_resize_(
    const at::Tensor& self,
    at::IntArrayRef size,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return at::native::resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = getMluTensorImpl(self);
  resize_impl_mlu_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    TORCH_CHECK(
        memory_format != at::MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  return self;
}

} // namespace ops
} // namespace torch_mlu
