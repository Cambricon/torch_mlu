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

#pragma once
#include "framework/core/device.h"
#include "framework/core/mlu_guard.h"
#include "framework/core/MLUStream.h"
#include "utils/Export.h"
#include "cnrt.h" // NOLINT

namespace torch_mlu {
struct TORCH_MLU_API MLUEvent {
  MLUEvent() {}
  MLUEvent(unsigned int flags) : flags_{flags} {}

  MLUEvent(DeviceIndex device_index, const cnrtIpcNotifierHandle* handle) {
    device_index_ = device_index;
    torch_mlu::mlu::MLUGuard guard(device_index_);
    TORCH_CNRT_CHECK(cnrtIpcOpenNotifierHandle(&event_, *handle));
    is_created_ = true;
  }

  ~MLUEvent() {
    if (is_created_) {
      destroyMLUEvent();
    }
  }

  MLUEvent(const MLUEvent&) = delete;
  MLUEvent& operator=(const MLUEvent&) = delete;

  MLUEvent(MLUEvent&& other) {
    moveHelper(std::move(other));
  }
  MLUEvent& operator=(MLUEvent&& other) {
    moveHelper(std::move(other));
    return *this;
  }

  // Less than operator (to allow use in sets)
  friend bool operator<(const MLUEvent& left, const MLUEvent& right) {
    return left.event_ < right.event_;
  }

  bool isCreated() const {
    return is_created_;
  }
  c10::DeviceIndex device_index() const {
    return device_index_;
  }
  cnrtNotifier_t event() {
    return event_;
  }

  std::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kPrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  void place(const MLUStream& stream);

  void place() {
    place(getCurrentMLUStream());
  }

  void placeOnce(const MLUStream& stream);

  float elapsed_time(const MLUEvent& other) const;

  float hardware_time(const MLUEvent& other) const;

  void wait(const MLUStream& stream);

  bool query() const;

  void synchronize();

  void ipc_handle(cnrtIpcNotifierHandle* handle);

 private:
  unsigned int flags_ = CNRT_NOTIFIER_DISABLE_TIMING_ALL;
  int device_index_ = -1;
  cnrtNotifier_t event_;
  bool is_created_ = false;
  bool was_placed_ = false;

  void moveHelper(MLUEvent&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(was_placed_, other.was_placed_);
    std::swap(event_, other.event_);
    std::swap(device_index_, other.device_index_);
    std::swap(flags_, other.flags_);
  }

  void destroyMLUEvent() {
    torch_mlu::mlu::MLUGuard guard(device_index_);
    TORCH_CNRT_CHECK(cnrtNotifierDestroy(event_));
  }

  void createMLUEvent(DeviceIndex device_index) {
    device_index_ = device_index;
    torch_mlu::mlu::MLUGuard guard(device_index_);
    TORCH_CNRT_CHECK(cnrtNotifierCreateWithFlags(&event_, flags_));
    is_created_ = true;
  }
};

} // namespace torch_mlu
