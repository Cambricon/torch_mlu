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

#include "framework/core/guard_impl.h"
#include "framework/core/device_utils.h"
#include "framework/core/caching_allocator.h"

namespace torch_mlu {
namespace mlu {

constexpr at::DeviceType MLUGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(PrivateUse1, MLUGuardImpl);

#define REGISTER_PRIVATEUSE1_BACKEND(name)    \
  int rename_privateuse1_backend() {          \
    c10::register_privateuse1_backend(#name); \
    return 0;                                 \
  }                                           \
  static const int _temp_##name = rename_privateuse1_backend();

REGISTER_PRIVATEUSE1_BACKEND(mlu)

c10::Device MLUGuardImpl::exchangeDevice(c10::Device device) const {
  AT_ASSERT(device.type() == at::DeviceType::PrivateUse1);
  c10::Device old_device = getDevice();
  if (old_device.index() != device.index()) {
    setDevice(device);
  }
  return old_device;
}

void MLUGuardImpl::setDevice(c10::Device device) const {
  TORCH_INTERNAL_ASSERT(device.is_privateuseone());
  torch_mlu::setDevice(device.index());
}

void MLUGuardImpl::uncheckedSetDevice(c10::Device device) const noexcept {
  torch_mlu::setDevice(device.index());
}

void MLUGuardImpl::record(
    void** event,
    const c10::Stream& stream,
    const c10::DeviceIndex device_index,
    const c10::EventFlag flag) const {
  TORCH_CHECK(
      device_index == -1 || device_index == stream.device_index(),
      "Event device index ",
      device_index,
      " does not match recording stream's device index ",
      stream.device_index(),
      ".");

  cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(*event);
  MLUStream mlu_stream(stream);
  cnrtQueue_t mlu_queue = mlu_stream.stream();

  // Moves to stream's device to record
  const auto orig_device = getDevice();
  setDevice(stream.device());

  // Create the Notifier
  if (!mlu_event)
    createEvent(&mlu_event, flag);
  TORCH_CNRT_CHECK(cnrtPlaceNotifier(mlu_event, mlu_queue));
  *event = mlu_event;

  // Resets device
  setDevice(orig_device);
}

void MLUGuardImpl::block(void* event, const c10::Stream& stream) const {
  if (!event)
    return;
  cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(event);
  MLUStream mlu_stream(stream);
  const auto orig_device = getDevice();
  setDevice(stream.device());
  TORCH_CNRT_CHECK(cnrtQueueWaitNotifier(mlu_event, mlu_stream.stream(), 0));
  setDevice(orig_device);
}

void MLUGuardImpl::recordDataPtrOnStream(
    const c10::DataPtr& data_ptr,
    const c10::Stream& stream) const {
  MLUStream mlu_stream{stream};
  torch_mlu::MLUCachingAllocator::recordStream(data_ptr, mlu_stream);
}

} // namespace mlu
} // namespace torch_mlu
