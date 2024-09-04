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

#include "framework/core/MLUStream.h"
#include "framework/hooks/MLUHooks.h"

namespace torch_mlu {

using CaptureId_t = unsigned long;

// first is set if the instance is created by MLUGraph::capture_begin.
// second is set if the instance is created by torch_mlu::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

// RAII guard for "cnrtQueueCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct MLUQueueCaptureModeGuard {
  MLUQueueCaptureModeGuard(cnrtQueueCaptureMode desired) {
    strictness_ = desired;
    TORCH_CNRT_CHECK(cnrtThreadExchangeQueueCaptureMode(&strictness_));
  }
  ~MLUQueueCaptureModeGuard() {
    TORCH_CNRT_CHECK(cnrtThreadExchangeQueueCaptureMode(&strictness_));
  }

 private:
  cnrtQueueCaptureMode strictness_;
};

// Protects against enum cnrtQueueCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(
    int(cnrtQueueCaptureStatus::cnrtQueueCaptureStatusNone) == 0,
    "unexpected int(cnrtQueueCaptureStatusNone) value");
static_assert(
    int(cnrtQueueCaptureStatus::cnrtQueueCaptureStatusActive) == 1,
    "unexpected int(cnrtQueueCaptureStatusActive) value");
static_assert(
    int(cnrtQueueCaptureStatus::cnrtQueueCaptureStatusInvalidated) == 2,
    "unexpected int(cnrtQueueCaptureStatusInvalidated) value");

enum class CaptureStatus : int {
  None = int(cnrtQueueCaptureStatus::cnrtQueueCaptureStatusNone),
  Active = int(cnrtQueueCaptureStatus::cnrtQueueCaptureStatusActive),
  Invalidated = int(cnrtQueueCaptureStatus::cnrtQueueCaptureStatusInvalidated)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "cnrtQueueCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "cnrtQueueCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "cnrtQueueCaptureStatusInvalidated";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown MLU graph CaptureStatus", int(status));
  }
  return os;
}

// Use this version where you're sure a MLU context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  cnrtQueueCaptureStatus is_capturing;
  TORCH_CNRT_CHECK(cnrtQueueIsCapturing(getCurrentMLUStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

inline CaptureStatus currentStreamCaptureStatus() {
  // don't create a context if we don't have to
  if (hasPrimaryContext(current_device())) {
    return currentStreamCaptureStatusMayInitCtx();
  } else {
    return CaptureStatus::None;
  }
}

inline void assertNotCapturing(std::string attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(
      status == CaptureStatus::None,
      attempt,
      " during MLU graph capture. If you need this call to be captured, "
      "please file an issue. "
      "Current cnrtQueueCaptureStatus: ",
      status);
}

} // namespace torch_mlu
