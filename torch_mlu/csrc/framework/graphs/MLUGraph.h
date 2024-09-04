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

#include <ATen/Tensor.h>
#include <c10/core/Device.h>

#include "framework/graphs/MLUGraphUtils.h"

namespace torch_mlu {

struct MLUGeneratorImpl;

// Standalone way to get a unique mempool id usable as a pool=... argument
// to MLUGraph::capture_begin
TORCH_MLU_API MempoolId_t graph_pool_handle();

struct TORCH_MLU_API MLUGraph {
  MLUGraph();
  ~MLUGraph();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  void capture_begin(
      MempoolId_t pool = {0, 0},
      cnrtQueueCaptureMode capture_mode = cnrtQueueCaptureModeGlobal);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

 protected:
  cnrtTaskTopo_t graph_ = nullptr;
  cnrtTaskTopoEntity_t graph_exec_ = nullptr;

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if cnrtQueueEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by
  // cnrtTaskTopoInstantiate to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if cnrtTaskTopoInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, used to
  // specify the pool.
  CaptureId_t id_;

  // the ID assigned by cuda during graph capture,
  // used to identify when a stream is participating in capture
  CaptureId_t capture_id_ = -1;

  // uuid used to request a particular private mempool from
  // MLUCachingAllocator. By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's
  // mempool_id_ will be set to the other graph's mempool_id_, and therefore
  // share a mempool with the other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from
  // graph_pool_handle(), it will share a mempool with any other captures that
  // used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // MLUStream on which capture began
  torch_mlu::MLUStream capture_stream_;

  // Default generator on device where capture began
  MLUGeneratorImpl* capture_gen_;

  // Device where capture occurred. Right now, for simplicity, we require all
  // ops in a capture to run on the same device.
  int capture_dev_;

  // RNG state trackers
  at::Tensor seed_extragraph_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

} // namespace torch_mlu
