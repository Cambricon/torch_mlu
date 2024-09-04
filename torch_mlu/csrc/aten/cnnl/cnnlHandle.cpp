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

#include <memory>
// torch header
#include "ATen/cuda/detail/DeviceThreadHandles.h"

// torch_mlu header
#include "aten/cnnl/cnnlHandle.h"
#include "aten/utils/exceptions.h"
#include "framework/core/device.h"
#include "framework/core/MLUStream.h"

namespace torch_mlu {
namespace {

void createCnnlHandle(cnnlHandle_t* handle) {
  TORCH_CNNL_CHECK(cnnlCreate(handle));
}

void destroyCnnlHandle(cnnlHandle_t handle) {
  if (handle) {
    TORCH_CNNL_CHECK(cnnlDestroy(handle));
    handle = nullptr;
  }
}

using CnnlPoolType = at::cuda::
    DeviceThreadHandlePool<cnnlHandle_t, createCnnlHandle, destroyCnnlHandle>;
} // namespace

cnnlHandle_t getCurrentHandle(c10::DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = current_device();
  }

  static auto pool = std::make_shared<CnnlPoolType>();
  thread_local std::unique_ptr<CnnlPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());
  auto handle = myPoolWindow->reserve(static_cast<int>(device_index));
  TORCH_CNNL_CHECK(
      cnnlSetQueue(handle, getCurrentMLUStream(device_index).stream()));
  return handle;
}

} // namespace torch_mlu
