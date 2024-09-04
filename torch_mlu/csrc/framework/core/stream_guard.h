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

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "aten/utils/exceptions.h"
#include "framework/core/device.h"
#include "framework/core/MLUStream.h"
#include "framework/core/guard_impl.h"
#include "utils/common.h"
#include "cnrt.h" // NOLINT

using torch_mlu::MLUStream;

namespace torch_mlu {
namespace mlu {

struct MLUStreamGuard {
  MLUStreamGuard() = delete;

  explicit MLUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  MLUStreamGuard(const MLUStreamGuard&) = delete;
  MLUStreamGuard& operator=(const MLUStreamGuard&) = delete;

  /// Move is disallowed, as MLUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  MLUStreamGuard(MLUStreamGuard&& other) = delete;
  MLUStreamGuard& operator=(MLUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a MLU MLUStream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the MLU MLUStream that was set at the time the guard was
  /// constructed.
  MLUStream original_stream() const {
    return MLUStream(MLUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent MLU MLUStream that was set using this device
  /// guard, either from construction, or via set_stream.
  MLUStream current_stream() const {
    return MLUStream(MLUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent MLU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  c10::Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the MLU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  c10::Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<torch_mlu::mlu::MLUGuardImpl> guard_;
};

/// A variant of MLUMultiStreamGuard that is specialized for MLU.
struct MLUMultiStreamGuard {
  explicit MLUMultiStreamGuard(c10::ArrayRef<torch_mlu::MLUStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  MLUMultiStreamGuard(const MLUMultiStreamGuard&) = delete;
  MLUMultiStreamGuard& operator=(const MLUMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  MLUMultiStreamGuard(MLUMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  MLUMultiStreamGuard& operator=(MLUMultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<torch_mlu::mlu::MLUGuardImpl> guard_;

  static std::vector<c10::Stream> unwrapStreams(
      c10::ArrayRef<torch_mlu::MLUStream> mlu_streams) {
    std::vector<c10::Stream> streams;
    streams.reserve(mlu_streams.size());
    for (const torch_mlu::MLUStream& mlu_stream : mlu_streams) {
      streams.push_back(mlu_stream);
    }
    return streams;
  }
};

} // namespace mlu
} // namespace torch_mlu
