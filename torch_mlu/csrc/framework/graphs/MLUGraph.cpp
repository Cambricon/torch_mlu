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

#include <ATen/Functions.h>

#include "framework/core/caching_allocator.h"
#include "framework/graphs/MLUGraph.h"
#include "framework/generator/generator_impl.h"

namespace torch_mlu {

static bool _mlu_graphs_debug = false;

MempoolId_t graph_pool_handle() {
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by
  // graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created
  // from cnrtQueueGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
}

// Get the expected id of a capture sequence so that we can call
// beginAllocateQueueToPool before starting a graph capture
CaptureId_t capture_sequence_id() {
  // id starts at 1:
  // Ensures uuid count starts at 1. 0 is reserved to mean "not set by
  // cnrtQueueGetCaptureInfo".
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

MLUGraph::MLUGraph()
    // MLUStreams may not be default-constructed.
    : capture_stream_(getCurrentMLUStream()) {}

void MLUGraph::capture_begin(
    MempoolId_t pool /*=0*/,
    cnrtQueueCaptureMode capture_mode) {
  TORCH_CHECK(
      !has_graph_exec_,
      "This MLUGraph instance already owns a captured graph. "
      "To capture a new graph, create a new instance.");

  // For now, a MLUGraph instance only accommodates the default generator on
  // the device that's current when capture begins. If any op in the captured
  // region uses a non-default generator, or a generator on another device, the
  // offending generator will throw an error. These restrictions simplify
  // MLUGraph, but could be relaxed in the future: in principle, the underlying
  // MLU calls do permit cross-device ops to be captured.
  auto* gen = at::get_generator_or_default<MLUGeneratorImpl>(
      c10::nullopt, torch_mlu::getDefaultMLUGenerator());

  auto options = at::TensorOptions().device(at::kPrivateUse1).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  gen->capture_prologue(
      seed_extragraph_.data_ptr<int64_t>(),
      offset_extragraph_.mutable_data_ptr<int64_t>());

  auto stream = torch_mlu::getCurrentMLUStream();

  TORCH_CHECK(
      stream != torch_mlu::getDefaultMLUStream(),
      "MLU graphs must be captured on a non-default stream. "
      "(However, after capture, it's ok to replay them on the "
      "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = torch_mlu::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be
    // nonzero. If pool was created by graph_pool_handle, second should be
    // nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our
    // mempool_id_. Sets just the first value, to distinguish it from
    // MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateQueueToPool is now called before
  // cnrtQueueBeginCapture to prevent an autograd thread's free() call
  // triggering an invalid mluEventRecord in the caching allocator due to the
  // capture status being updated _after_ a capture had already started.
  torch_mlu::MLUCachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, [this](cnrtQueue_t stream) {
        cnrtQueueCaptureStatus status;
        CaptureId_t stream_capture_id;
        TORCH_CNRT_CHECK(cnrtQueueGetCaptureInfo(
            stream, &status, &stream_capture_id, nullptr, nullptr, nullptr));
        return status == cnrtQueueCaptureStatus::cnrtQueueCaptureStatusActive &&
            stream_capture_id == capture_id_;
      });

  // cnrtQueueCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe MLU API calls during capture.
  TORCH_CNRT_CHECK(cnrtQueueBeginCapture(capture_stream_, capture_mode));

  cnrtQueueCaptureStatus status;
  TORCH_CNRT_CHECK(cnrtQueueGetCaptureInfo(
      stream, &status, &capture_id_, nullptr, nullptr, nullptr));
  TORCH_INTERNAL_ASSERT(
      status == cnrtQueueCaptureStatus::cnrtQueueCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
}

void MLUGraph::capture_end() {
  auto stream = torch_mlu::getCurrentMLUStream();

  TORCH_CHECK(
      stream == capture_stream_,
      "Capture must end on the same stream it began on.");

  TORCH_CNRT_CHECK(cnrtQueueEndCapture(capture_stream_, &graph_));

  torch_mlu::MLUCachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  // In typical graph usage some tensors (e.g. the tensors used for graph IO)
  // are not freed between replays. For CUDA, If Pytorch compiles and runs with
  // a CUDA 11.4+ toolkit, there's a chance the allocator backend is
  // cudaMallocAsync. For MLU, we don't consider cnrtMallocAsync yet.
  TORCH_CNRT_CHECK(
      cnrtTaskTopoInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

  has_graph_exec_ = true;

  auto* gen = at::get_generator_or_default<MLUGeneratorImpl>(
      c10::nullopt, torch_mlu::getDefaultMLUGenerator());
  TORCH_CHECK(
      gen == capture_gen_,
      "Default MLU RNG generator on current device at capture end "
      "is different from default generator on current device "
      "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  size_t numMLUGraphNodes = 0;
  TORCH_CNRT_CHECK(cnrtTaskTopoGetNodes(graph_, nullptr, &numMLUGraphNodes));
  if (numMLUGraphNodes == 0) {
    TORCH_WARN(
        "The MLU Graph is empty. This usually means that the graph was ",
        "attempted to be captured on wrong device or stream.");
  }

  // check if debug path is set
  if (!_mlu_graphs_debug) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    TORCH_CNRT_CHECK(cnrtTaskTopoDestroy(graph_));
    has_graph_ = false;
  } else {
    TORCH_WARN(
        "DEBUG: TORCH_MLUGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
}

void MLUGraph::replay() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called MLUGraph::replay without a preceding successful capture.");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto* gen = at::get_generator_or_default<MLUGeneratorImpl>(
      c10::nullopt, torch_mlu::getDefaultMLUGenerator());
  PhiloxMLUState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_mlu_state(wholegraph_increment_);
  }
  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  // graph_exec_ may be replayed in any stream.
  TORCH_CNRT_CHECK(
      cnrtTaskTopoEntityInvoke(graph_exec_, getCurrentMLUStream()));
}

void MLUGraph::enable_debug_mode() {
  _mlu_graphs_debug = true;
}

void MLUGraph::debug_dump(const std::string& debug_path) {
  if (_mlu_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN(
          "DEBUG: calling cnrtTaskTopoDebugDotPrint() with ", debug_path);
      TORCH_CNRT_CHECK(cnrtTaskTopoDebugDotPrint(
          graph_, debug_path.c_str(), 1 << 0)); // most verbose output
      TORCH_CNRT_CHECK(cnrtTaskTopoDestroy(graph_));
      has_graph_ = false;
    }
  } else {
    TORCH_WARN("MLU Graphs debug not enabled");
  }
}

void MLUGraph::reset() {
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this
  // MLUGraph, the generator, and the allocator could end up in all kinds of
  // weird states depending where failure occurred. If the user catches the
  // failure exception in a script, or is running in REPL or (god forbid) a
  // Jupyter notebook, I don't see an easy way for reset() to gracefully fix all
  // such possible error states.
  if (has_graph_ || has_graph_exec_) {
    torch_mlu::MLUCachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    TORCH_CNRT_WARN(cnrtTaskTopoDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    TORCH_CNRT_WARN(cnrtTaskTopoEntityDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory
// pool as this graph.
MempoolId_t MLUGraph::pool() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called MLUGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

MLUGraph::~MLUGraph() {
  reset();
}

} // namespace torch_mlu
