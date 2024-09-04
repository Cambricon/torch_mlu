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

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "aten/utils/tensor_util.h"
#include "cncl.h" // NOLINT
#include "framework/core/MLUStream.h"
#include "framework/core/MLUEvent.h"
#include <c10/util/irange.h>
#include <c10/util/ApproximateClock.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>
#include <fstream>
#include "Utils.h"


#define C10D_CNCL_CHECK(cmd, failure_reason)                              \
  do {                                                                    \
    cnclResult_t error = cmd;                                             \
    if (error != CNCL_RET_SUCCESS) {                                      \
      std::string err = "CNCL error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + ", " +                               \
          getCnclErrorDetailStr(error, failure_reason);                   \
      TORCH_CHECK(false, err);                                            \
    }                                                                     \
  } while (0)

#define C10D_CNCL_ASSERT(cmd)                 \
  do {                                        \
    cnclResult_t res = cmd;                   \
    if (res != CNCL_RET_SUCCESS) {            \
      std::string err = cnclGetErrorStr(res); \
      fprintf(                                \
          stderr,                             \
          "CNCL error in: %s:%d, %s\n",       \
          __FILE__,                           \
          __LINE__,                           \
          err.c_str());                       \
      abort();                                \
    }                                         \
  } while (0)

// Provides additional detail into CNCL error codes based on when these are
// thrown in the CNCL codebase.
TORCH_MLU_API std::string getCnclErrorDetailStr(
    cnclResult_t error,
    std::optional<std::string> process_group_failure_reason = c10::nullopt);

namespace torch_mlu {

namespace cncl::detail {

void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream);

void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream);

void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    int32_t root = 0);

void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    int32_t root = 0);

} // namespace cncl::detail

TORCH_MLU_API std::unordered_map<std::string, MLUStream> getCnclStream(
    const DeviceIndex& device_index);

void updateCnclStream(const cnclCliqueId* cncl_id, MLUStream cncl_stream);

void clearCnclStream(const cnclCliqueId* cncl_id);

std::mutex* getFreeMutex();

/* Helper used by work::getDuration() and cncl flight recorder */
float getDurationFromEvent(
    torch_mlu::MLUEvent& cnclStartEvent,
    torch_mlu::MLUEvent& cnclEndEvent);

inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}

inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

inline std::string pickle_str(const c10::IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    torch::jit::Pickler pickler(
        writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}

// Write CNCL debug Info to local disk or any storage users define.
// There are some constrains we set for the debug info writer:
// 1. writer should only be registered once.
// 2. Once registered, users cannot change it including un-register.
// 3. It is recomended to register the customized writer in the trainer setup,
//    If users don't register before calling launchAsyncDebugDump, then users
//    lose the chance to register (and the default writer will be
//    auto-registered).
class TORCH_MLU_API DebugInfoWriter {
 public:
  virtual ~DebugInfoWriter();
  virtual void write(const std::string& cnclTrace);
  static DebugInfoWriter& getWriter(int rank);
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);

 protected:
  DebugInfoWriter(std::string namePrefix, int rank) {
    filename_ = c10::str(namePrefix, rank);
  }
  std::string filename_;

 private:
  static std::unique_ptr<DebugInfoWriter> writer_;
  static std::atomic<bool> hasWriterRegistered_;
};


struct CNCLTraceBuffer {
  static CNCLTraceBuffer* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static CNCLTraceBuffer* instance = new CNCLTraceBuffer();
    return instance;
  }
  CNCLTraceBuffer() {
    max_entries_ = getCvarInt(
        {"TORCH_CNCL_TRACE_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE"}, 0);
    capture_cpp_stack_ = getCvarBool(
        {"TORCH_CNCL_TRACE_CPP_STACK", "TORCH_NCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
    pg_name_to_ranks_ = {};
  }
  using Event = torch_mlu::MLUEvent;
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t pg_id_;
    size_t seq_id_; // as tracked by the process group
    const char* profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;
    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    Event *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_;
    std::optional<float> duration_;

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on MLU APIs.
    std::optional<c10::time_t> time_discovered_started_;

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually complated, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on MLU
    // APIs
    std::optional<c10::time_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int, 4> input_dims_;
    c10::SmallVector<int, 4> output_dims_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;
  std::map<std::string, std::vector<uint64_t>> pg_name_to_ranks_;

  std::optional<size_t> record(
      size_t pg_id,
      size_t seq_id,
      const char* profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      Event* start,
      Event* end) {
    if (!enabled_) {
      return c10::nullopt;
    }
    auto traceback =
        torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
    std::lock_guard<std::mutex> guard(mutex_);

    auto te = Entry{
        id_,
        pg_id,
        seq_id,
        profiling_name == nullptr ? "" : profiling_name,
        std::move(traceback),
        std::move(start),
        std::move(end),
        c10::getTime()};

    for (const auto& input : inputs) {
      c10::IntArrayRef sizes = input.sizes();
      te.input_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    for (const auto& output : outputs) {
      c10::IntArrayRef sizes = output.sizes();
      te.output_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    if (entries_.size() < max_entries_) {
      entries_.emplace_back(std::move(te));
    } else {
      entries_[next_++] = std::move(te);
      if (next_ == max_entries_) {
        next_ = 0;
      }
    }
    return id_++;
  }

  void record_pg_ranks(
      const std::string& pg_name,
      std::vector<uint64_t> ranks) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    pg_name_to_ranks_[pg_name] = ranks;
  }

  void update_state(Entry& r) {
    if (r.start_ != nullptr) {
      bool started = r.start_->query();
      if (started && !r.time_discovered_started_) {
        r.time_discovered_started_ = c10::getTime();
      }
    }
    if (r.end_ != nullptr) {
      bool completed = r.end_->query();
      if (completed && !r.time_discovered_completed_) {
        r.time_discovered_completed_ = c10::getTime();
      }
    }
  }

  std::vector<Entry> dump_entries() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<Entry> result;
    result.reserve(entries_.size());
    result.insert(result.end(), entries_.begin() + next_, entries_.end());
    result.insert(result.end(), entries_.begin(), entries_.begin() + next_);
    // query any remaining events
    for (auto& r : result) {
      update_state(r);
      r.start_ = r.end_ = nullptr;
    }
    return result;
  }

  /*
  Mark an Event as completed and free its events.

  This is called by the watchdog thread, and is asynchronous from the
  perspective of the main thread.

  compute_duration defaults to true since retire_id is only called in the
  watchdog thread, which is currently a place we call cuda APIs which may hang,
  but care should be taken to avoid computing duration in any function that must
  never hang. (timing must also be enabled for compute_duration - see
  TORCH_CNCL_ENABLE_TIMING).
  */
  void retire_id(std::optional<size_t> id, bool compute_duration = true) {
    if (!enabled_ || !id) {
      return;
    }

    bool can_compute_duration = false;
    Event* startEvent = nullptr;
    Event* endEvent = nullptr;
    std::optional<float> duration = c10::nullopt;

    std::unique_lock<std::mutex> guard(mutex_);

    auto& entry = entries_.at(*id % max_entries_);
    if (entry.id_ == *id) {
      update_state(entry);

      if (compute_duration) {
        can_compute_duration = entry.time_discovered_completed_.has_value() &&
            entry.start_ && entry.end_;
        startEvent = entry.start_;
        endEvent = entry.end_;
      }
    }

    if (can_compute_duration) {
      // Compute duration without without holding the lock, because
      // cudaEventDuration() can hang, and we need to acquire the lock before we
      // can dump(), which we never want to block.
      guard.unlock();
      duration = getDurationFromEvent(*startEvent, *endEvent);
      guard.lock();

      // Refresh the entry ref, see if it has been overwritten
      entry = entries_.at(*id % max_entries_);
      if (entry.id_ != *id) {
        LOG(INFO)
            << "retire_id abandoned for id " << *id
            << ", event was overwritten while waiting to compute duration.";
        return;
      }
      if (duration.has_value()) {
        entry.duration_ = duration.value();
      }
    }

    entry.retired_ = true;
    entry.start_ = entry.end_ = nullptr;
  }

  std::string dump() {
    auto result = dump_entries();
    auto entries = new_list();
    c10::IValue entries_key = "entries";
    c10::IValue version_key = "version";
    // Update whenever changing contents or formatting of the dump
    // (minor when adding fields, major when changing existing fields)
    c10::IValue version_val = "1.1";

    c10::IValue pg_id_key = "pg_id";
    c10::IValue seq_id_key = "seq_id";
    c10::IValue profiling_name_key = "profiling_name";
    c10::IValue input_sizes_key = "input_sizes";
    c10::IValue output_sizes_key = "output_sizes";
    c10::IValue time_created_key = "time_created_ns";
    c10::IValue duration_key = "duration_ms";

    c10::IValue frames_key = "frames";
    c10::IValue state_key = "state";
    c10::IValue line_key = "line";
    c10::IValue name_key = "name";
    c10::IValue filename_key = "filename";
    c10::IValue retired_key = "retired";
    c10::IValue time_discovered_started_key = "time_discovered_started_ns";
    c10::IValue time_discovered_completed_key = "time_discovered_completed_ns";

    std::vector<torch::CapturedTraceback*> tracebacks;
    for (auto& e : result) {
      tracebacks.push_back(e.traceback_.get());
    }
    torch::SymbolizedTracebacks stracebacks = torch::symbolize(tracebacks);
    std::vector<c10::IValue> all_frames;
    for (const auto& f : stracebacks.all_frames) {
      auto d = new_dict();
      d.insert(name_key, f.funcname);
      d.insert(filename_key, f.filename);
      d.insert(line_key, int64_t(f.lineno));
      all_frames.emplace_back(std::move(d));
    }

    for (auto i : c10::irange(result.size())) {
      auto& e = result.at(i);
      auto& tb = stracebacks.tracebacks.at(i);
      auto dict = new_dict();
      dict.insert(pg_id_key, int64_t(e.pg_id_));
      dict.insert(seq_id_key, int64_t(e.seq_id_));
      dict.insert(profiling_name_key, e.profiling_name_);
      dict.insert(time_created_key, int64_t(e.time_created_));
      if (e.duration_) {
        dict.insert(duration_key, *e.duration_);
      }

      auto it = e.sizes_.begin();
      auto read_sizes = [&](const c10::SmallVector<int, 4>& dims) {
        auto sizes = new_list();
        for (auto dim : dims) {
          auto arg_sizes = new_list();
          for (auto i : c10::irange(dim)) {
            (void)i;
            arg_sizes.push_back(*it++);
          }
          sizes.push_back(arg_sizes);
        }
        return sizes;
      };

      dict.insert(input_sizes_key, read_sizes(e.input_dims_));
      dict.insert(output_sizes_key, read_sizes(e.output_dims_));
      if (e.time_discovered_completed_.has_value()) {
        dict.insert(state_key, "completed");
      } else if (e.time_discovered_started_.has_value()) {
        dict.insert(state_key, "started");
      } else {
        dict.insert(state_key, "scheduled");
      }

      dict.insert(
          time_discovered_started_key,
          e.time_discovered_started_.has_value()
              ? int64_t(*e.time_discovered_started_)
              : c10::IValue());
      dict.insert(
          time_discovered_completed_key,
          e.time_discovered_completed_.has_value()
              ? int64_t(*e.time_discovered_completed_)
              : c10::IValue());
      dict.insert(retired_key, e.retired_);

      auto frames = new_list();
      for (int64_t frame : tb) {
        frames.push_back(all_frames.at(frame));
      }
      dict.insert(frames_key, frames);
      entries.push_back(dict);
    }

    auto dict = new_dict();
    dict.insert(entries_key, entries);
    dict.insert(version_key, version_val);

    return pickle_str(dict);
  }
};

} // namespace torch_mlu
