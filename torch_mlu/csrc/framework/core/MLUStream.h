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

#include <c10/core/DeviceGuard.h>
#include <c10/util/CallOnce.h>
#include <c10/core/Stream.h>
#include "framework/core/device.h"
#include "utils/Export.h"
#include "cnrt.h" // NOLINT

using c10::Device;
using c10::DeviceIndex;
using c10::DeviceType;

namespace torch_mlu {

static constexpr int max_compile_time_stream_priorities = 4;

/**
 * Represents an abstract object of the MLU MLUStream, which comes with some
 * encapsulation and additional functionalityfor cnrtQueue_t.
 * cnrtQueue_t lifecycle is created and destory by the internal abstract
 * class MLUStreamInternals.
 */
class TORCH_MLU_API MLUStream {
 public:
  enum Unchecked { UNCHECKED };
  // MLUStream() {}
  explicit MLUStream(c10::Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == c10::DeviceType::PrivateUse1);
  }

  explicit MLUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

  bool operator==(const MLUStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const MLUStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  /// Implicit conversion to cnrtQueue_t.
  operator cnrtQueue_t() const {
    return stream();
  }

  /// Implicit conversion to Stream (a.k.a., forget that the stream is a
  /// cnrtQueue).
  operator c10::Stream() const {
    return unwrap();
  }

  c10::DeviceType device_type() const {
    return DeviceType::PrivateUse1;
  }

  /// Get the MLU device index that this stream is associated with.
  c10::DeviceIndex device_index() const {
    return stream_.device_index();
  }

  /// Get the full Device that this stream is associated with.  The Device
  /// is guaranteed to be a MLU device.
  c10::Device device() const {
    return c10::Device(c10::DeviceType::PrivateUse1, device_index());
  }

  /// Return the stream ID corresponding to this particular stream.
  c10::StreamId id() const {
    return stream_.id();
  }

  bool query() const {
    c10::DeviceGuard guard{stream_.device()};
    cnrtRet_t err = cnrtQueueQuery(stream());
    if (err == CNRT_RET_SUCCESS) {
      return true;
    } else if (err != cnrtErrorNotReady) {
      TORCH_CNRT_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)cnrtGetLastError();
    }
    return false;
  }

  void synchronize() const {
    c10::DeviceGuard guard{stream_.device()};
    TORCH_CNRT_CHECK(cnrtQueueSync(stream()));
  }

  int priority() const {
    // TORCH_INTERNAL_ASSERT(0, "now, priority() function is not supported.");
    c10::DeviceGuard guard{stream_.device()};
    int priority = 0;
    TORCH_CNRT_CHECK(cnrtQueueGetPriority(stream(), &priority));
    // Currently Catch support 0,1,2,3, and transfer -3,-2,-1,0 to follow native
    // Pytorch. external stream may come from third lib, it's priority can't be
    // confirmed. external Stream return real mlu priority, which it's priority
    // out of [0, 3].
    if (priority >= 0 && priority <= 3) {
      return priority - 3;
    }
    return priority;
  }

  // Explicit conversion to cnrtQueue_t.
  cnrtQueue_t stream() const;

  // Explicit conversion to Stream.
  c10::Stream unwrap() const {
    return stream_;
  }

  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  static MLUStream unpack3(
      c10::StreamId stream_id,
      DeviceIndex device_index,
      DeviceType device_type) {
    return MLUStream(
        c10::Stream::unpack3(stream_id, device_index, device_type));
  }

  static std::tuple<int, int> priority_range() {
    // Note: this returns the range of priority **supported by PyTorch**, not
    // the range of priority **supported by MLU** is [7, 0]. The former is a
    // subset of the latter. Currently Catch supports 0,1,2,3, and tansfer to
    // -3,-2,-1,0.
    int least_priority, greatest_priority;
    TORCH_CNRT_CHECK(
        cnrtDeviceGetQueuePriorityRange(&least_priority, &greatest_priority));

    TORCH_INTERNAL_ASSERT(
        least_priority == 7, "Unexpected MLU stream priority range");
    TORCH_INTERNAL_ASSERT(
        greatest_priority == 0, "Unexpected MLU stream priority range");
    // Get a subset of priority range is [0, -3] to follow native Pytorch.
    return std::make_tuple(0, -3);
  }

 private:
  c10::Stream stream_;
};

/**
 * Get a new Stream from the MLU Stream pool.  You can think of this
 * as "creating" a new Stream, but no such creation actually happens;
 * instead, Streams are preallocated from the pool and returned in a
 * round-robin fashion.
 */

TORCH_MLU_API MLUStream
getStreamFromPool(const int priority, DeviceIndex device_index = -1);
TORCH_MLU_API MLUStream getStreamFromPool(
    const bool isHighPriority = false,
    DeviceIndex device_index = -1);

/**
 * Get a MLUStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
TORCH_MLU_API MLUStream
getStreamFromExternal(cnrtQueue_t ext_stream, DeviceIndex device_index);

/**
 * Get the default MLU Stream, for the passed MLU device, or for the
 * current device if no device index is passed.  The default stream is
 * where most computation occurs when you aren't explicitly using
 * streams.
 */
TORCH_MLU_API MLUStream getDefaultMLUStream(DeviceIndex device_index = -1);

/**
 * Get the current MLU Stream, for the passed MLU device, or for the
 * current device if no device index is passed.  The current MLU Stream
 * will usually be the default MLU Stream for the device, but it may
 * be different if someone called 'setCurrentMLUStream' or used 'StreamGuard'
 * or 'MLUStreamGuard'.
 */
TORCH_MLU_API MLUStream getCurrentMLUStream(DeviceIndex device_index = -1);

/**
 * Get the current cnrtQueue_t from getCurrentMLUStream.
 */
TORCH_MLU_API cnrtQueue_t getCurMLUStream(DeviceIndex device_index = -1);

/**
 * Set the current Stream on the device of the passed in Stream to be
 * the passed in Stream.  Yes, you read that right: this function
 * has *nothing* to do with the current device: it toggles the current
 * Stream of the device of the passed Stream.
 *
 * Confused?  Avoid using this function; prefer using 'MLUStreamGuard' instead
 * (which will switch both your current device and current Stream in the way you
 * expect, and reset it back to its original state afterwards).
 */
TORCH_MLU_API void setCurrentMLUStream(MLUStream stream);

TORCH_MLU_API std::ostream& operator<<(
    std::ostream& stream,
    const MLUStream& mlu_stream);

} // namespace torch_mlu

namespace std {
template <>
struct hash<torch_mlu::MLUStream> {
  size_t operator()(torch_mlu::MLUStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
