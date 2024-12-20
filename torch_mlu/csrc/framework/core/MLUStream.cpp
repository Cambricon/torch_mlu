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

#include "framework/core/MLUStream.h"
#include <atomic>
#include <mutex>
#include <thread>

namespace torch_mlu {

// Global stream state and constants
static c10::once_flag init_flag;
static c10::DeviceIndex num_mlus = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr int kStreamTypeBits = 4;

static int max_stream_priorities;

// Non-default streams
// Note: the number of MLU devices is determined at run time,
// and the low and high priority pools are lazily initialized
// when the first stream is requested for a device.
// The device flags track the initialization of each device, while
// the low and high priority counters track, for each device, the next stream
// in the pool to be returned when a stream is requested (round-robin fashion
// , see the note in MLUStream.h).
// The streams are "leaked": they are created but never destroyed because the
// destruction of global variables could happen after the CNRT has
// already been destroyed and thus invoking cnrtQueueDestroy could lead to a
// crash. It's likely an issue in MLU, but to be safe - let's just "forget"
// the destruction.

static c10::once_flag device_flags[MLU_DEVICE_NUM_MAX];

static std::atomic<uint32_t>
    priority_counters[max_compile_time_stream_priorities][MLU_DEVICE_NUM_MAX];

static cnrtQueue_t streams[max_compile_time_stream_priorities]
                          [MLU_DEVICE_NUM_MAX][kStreamsPerPool];

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 54 bits --  -- 5 bits -----  -- 4 bits --     --1 bit --
// zeros          stream id index  StreamIdType     Ext/native stream
//                ignored for ext   ignored for ext
// for external stream, StreamID is a cnrtQueue_t pointer
// this means that last bit will always be 0
// so when constructing StreamId for a native stream we set last bit to 1
// to distinguish between native and external streams
//
//
// We are obligated to treat the stream ID 0 as the default stream, per the
// invariant specified in c10::Stream, so this is one exception to
// "last bit = 1 for native streams". However, all other numbers are entirely
// an internal implementation detail, we reserve the right to renumber streams
// however we like.
//
// Note that it is really important that the MSB is zero; StreamId is a
// *signed* integer, and unsigned to signed conversion outside of the
// bounds of signed integer representation is undefined behavior.  You
// could work around this with something like
// https://stackoverflow.com/questions/13150449/efficient-unsigned-to-signed-cast-avoiding-implementation-defined-behavior
// but it seems a bit overkill for this.
//
// Also, external managed stream pointers (cudaStream_t) can be directly stored
// in the Id field so in this case, we need to check the stream alignment.

class StreamIdType {
  // StreamIdType encodes whether this stream is DEFAULT, EXTernal or
  // for all other native streams, the stream priority (higher value is higher
  // priority)
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

 public:
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExt() const {
    return EXT == stream_type;
  }

  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    return stream_type;
  }
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << int(s.getStreamType());
  }
  return stream;
}

static inline StreamIdType streamIdType(c10::StreamId s) {
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto val = (s >> 1) & mask_for_type;
  TORCH_INTERNAL_ASSERT(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

static inline size_t streamIdIndex(c10::StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

c10::StreamId makeStreamId(StreamIdType qt, size_t qi) {
  if (qt.isDefault()) {
    return static_cast<c10::StreamId>(0);
  }
  return (static_cast<c10::StreamId>(qi) << (kStreamTypeBits + 1)) |
      static_cast<c10::StreamId>(qt.getStreamType() << 1 | 1);
}

// Thread-local current streams
static thread_local std::unique_ptr<c10::StreamId[]> current_streams = nullptr;

// Populates global values.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  num_mlus = device_count();
  // Check if the number of MLUs matches the expected compile-time max number
  // of MLUs.
  TORCH_CHECK(
      num_mlus <= MLU_DEVICE_NUM_MAX,
      "Number of MLU devices on the machine is larger than the compiled "
      "max number of mlus expected (",
      MLU_DEVICE_NUM_MAX,
      "). Increase that and recompile.");
  int leastPriority = -1, greatestPriority = -1;
  TORCH_CNRT_CHECK(
      cnrtDeviceGetQueuePriorityRange(&leastPriority, &greatestPriority);)
  auto range = leastPriority - greatestPriority + 1;
  max_stream_priorities = range >= max_compile_time_stream_priorities
      ? max_compile_time_stream_priorities
      : range;
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  c10::DeviceGuard device_guard{
      c10::Device(c10::DeviceType::PrivateUse1, device_index)};

  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_stream_priorities)) {
      auto& stream = streams[p][device_index][i];
      auto pri = max_stream_priorities - 1 - p;
      TORCH_CNRT_CHECK(cnrtQueueCreateWithPriority(&stream, 0, pri));
      priority_counters[p][device_index] = 0;
    }
  }
}

// Init front-end to ensure initialization only occurs once
static void initMLUStreamsOnce() {
  // Inits default streams (once, globally)
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams = std::make_unique<c10::StreamId[]>(num_mlus);
  for (const auto i : c10::irange(num_mlus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

// Helper to verify the MLU index is valid
static inline void check_mlu(c10::DeviceIndex device_index) {
  AT_ASSERT(device_index >= 0 && device_index < num_mlus);
}

// Helper to determine the index of the stream to return
// Note: Streams are returned round-robin (see note in stream.h)
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

MLUStream MLUStreamForId(DeviceIndex device_index, c10::StreamId stream_id) {
  return MLUStream(
      MLUStream::UNCHECKED,
      c10::Stream(
          c10::Stream::UNSAFE,
          c10::Device(c10::DeviceType::PrivateUse1, device_index),
          stream_id));
}

// See Note [StreamId assignment]
cnrtQueue_t MLUStream::stream() const {
  c10::DeviceIndex device_index = stream_.device_index();
  c10::StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  if (st.isDefault()) {
    TORCH_INTERNAL_ASSERT(
        si == 0,
        "Unrecognized stream ",
        stream_,
        " (I think this should be the default stream, but I got a non-zero index ",
        si,
        ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like torch_mlu::getStreamFromPool() to get a new stream.");
    // cndrv2.x, default stream is per-thread stream,
    // cndrv>=3.x, default stream is legacy stream.
    // We return the legacy stream as default stream
    // forcely to make the behavior same with cuda.
    return cnrtQueueLegacy;
  } else if (st.isExt()) {
    return reinterpret_cast<cnrtQueue_t>(stream_id);
  } else {
    auto streamType = st.getStreamType();
    TORCH_INTERNAL_ASSERT(
        streamType >= 1 && streamType <= max_stream_priorities,
        "Unrecognized stream ",
        stream_,
        " (I didn't recognize the stream type, ",
        st,
        " with the value ",
        streamType,
        ")");
    return streams[st.getStreamType() - 1][device_index][si];
  }
}

// Returns a stream from the requested pool
// Note: when called the first time on a device, this will create the
// stream pools for that device.
MLUStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  initMLUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }

  TORCH_CHECK(
      priority <= 0,
      "Expected mlu stream priority to be less than or equal to 0, got ",
      priority);

  check_mlu(device_index);

  // Initializes the stream pools (once)
  c10::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  auto pri_idx = -priority;
  pri_idx = std::min(pri_idx, max_stream_priorities - 1);
  const auto idx = get_idx(priority_counters[pri_idx][device_index]);
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return MLUStreamForId(device_index, makeStreamId(id_type, idx));
}

MLUStream getStreamFromPool(
    const bool isHighPriority,
    DeviceIndex device_index) {
  initMLUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  int priority = isHighPriority ? -max_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device_index);
}

MLUStream getStreamFromExternal(
    cnrtQueue_t ext_stream,
    DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return MLUStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}

MLUStream getDefaultMLUStream(DeviceIndex device_index) {
  initMLUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_mlu(device_index);
  return MLUStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

MLUStream getCurrentMLUStream(DeviceIndex device_index) {
  initMLUStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
  }
  check_mlu(device_index);
  return MLUStreamForId(device_index, current_streams[device_index]);
}

cnrtQueue_t getCurMLUStream(DeviceIndex device_index) {
  return getCurrentMLUStream(device_index).stream();
}

void setCurrentMLUStream(MLUStream stream) {
  initMLUStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const MLUStream& s) {
  return stream << s.unwrap();
}

} // namespace torch_mlu
