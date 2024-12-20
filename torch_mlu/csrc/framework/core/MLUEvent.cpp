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

#include "framework/core/MLUEvent.h"
#include "framework/core/MLUStream.h"

namespace torch_mlu {

void MLUEvent::place(const MLUStream& stream) {
  if (!is_created_) {
    createMLUEvent(stream.device_index());
  }
  TORCH_CHECK(
      device_index_ == stream.device_index(),
      "MLUEvent device ",
      device_index_,
      " does not match placing stream's device ",
      stream.device_index(),
      ".");
  torch_mlu::mlu::MLUGuard guard(device_index_);
  TORCH_CNRT_CHECK(cnrtPlaceNotifier(event_, stream.stream()));
  was_placed_ = true;
}

void MLUEvent::placeOnce(const MLUStream& stream) {
  if (!was_placed_)
    place(stream);
}

float MLUEvent::elapsed_time(const MLUEvent& other) const {
  TORCH_CHECK(
      is_created_ && other.isCreated(),
      "Both notifiers must be placed before calculating elapsed time.");
  float time_ms = 0;
  // See https://github.com/pytorch/pytorch/pull/122538
  torch_mlu::mlu::MLUGuard guard(device_index_);
  TORCH_CNRT_CHECK(cnrtNotifierElapsedTime(event_, other.event_, &time_ms));
  return time_ms;
}

float MLUEvent::hardware_time(const MLUEvent& other) const {
  TORCH_CHECK(
      is_created_ && other.isCreated(),
      "Both notifiers must be placed before calculating hardware time.");
  float time_us = 0;
  TORCH_CNRT_CHECK(cnrtNotifierDuration(event_, other.event_, &time_us));
  return time_us;
}

void MLUEvent::synchronize() {
  if (is_created_) {
    TORCH_CNRT_CHECK(cnrtWaitNotifier(event_));
  }
}

// set MLUGuard before using this interface.
bool MLUEvent::query() const {
  if (!is_created_) {
    return true;
  }
  cnrtRet_t err = cnrtQueryNotifier(event_);
  if (err == cnrtSuccess) {
    return true;
  } else if (err != cnrtErrorNotReady) {
    TORCH_CNRT_CHECK(err);
  }
  return false;
}

void MLUEvent::wait(const MLUStream& stream) {
  if (is_created_) {
    torch_mlu::mlu::MLUGuard guard(stream.device_index());
    TORCH_CNRT_CHECK(cnrtQueueWaitNotifier(event_, stream.stream(), 0));
  }
}

void MLUEvent::ipc_handle(cnrtIpcNotifierHandle* handle) {
  if (!is_created_) {
    // this MLUEvent object was initially constructed from flags but event_
    // is not created yet.
    createMLUEvent(getCurrentMLUStream().device_index());
  }
  torch_mlu::mlu::MLUGuard guard(device_index_);
  TORCH_CNRT_CHECK(cnrtIpcGetNotifierHandle(handle, event_));
}

} // namespace torch_mlu
