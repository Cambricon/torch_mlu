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

#include <pybind11/chrono.h>
#include <torch/extension.h>
#include <unordered_map>

#include "cncl.h"
#include "framework/core/MLUEvent.h"
#include "aten/utils/tensor_util.h"

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

namespace torch_mlu {

// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
constexpr const char* CNCL_BLOCKING_WAIT = "CNCL_BLOCKING_WAIT";

constexpr const char* CNCL_BACKEND_NAME = "cncl";

// Environment variable which controls whether or not we perform Async Error
// Handling with CNCL.
constexpr const char* CNCL_ASYNC_ERROR_HANDLING = "CNCL_ASYNC_ERROR_HANDLING";

// Environment Variable to control whether Desync Debug is enabled.
// This variable must be set together with NCCL_ASYNC_ERROR_HANDLING.
constexpr const char* CNCL_DESYNC_DEBUG = "CNCL_DESYNC_DEBUG";

// TearDown mode: tear down process upon error, see `WorkCNCL::handleCNCLGuard`
// Soft mode: just clean up collectives and abort communicators without tearing
// down process
enum ErrorHandlingMode { NoHandling = 0, TearDown = 1, CleanUpOnly = 2 };

// If set, ProcessGroupCNCL doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// Instead, it stashes live references to those tensors until after
// user-facing streams are synced with comm streams.
// See stashed_for_allocator_safety_ below.
constexpr const char* TORCH_CNCL_AVOID_RECORD_STREAMS =
    "TORCH_CNCL_AVOID_RECORD_STREAMS";

// RAII wrapper for CNCL communicator in a process
class TORCH_MLU_API CNCLComm {
 public:
  explicit CNCLComm(cnclComm_t cnclComm) // NOSONAR
      : cncl_comm_(cnclComm),
        aborted_(false),
        cncl_async_err_(CNCL_RET_SUCCESS),
        comm_failure_reason_(c10::nullopt) {}

  CNCLComm() : CNCLComm(nullptr) {}

  ~CNCLComm() noexcept;

  static std::shared_ptr<CNCLComm> create(
      int numRanks,
      int rank,
      int device,
      const cnclCliqueId_t clique_id);

  // Must not be copyable
  CNCLComm(const CNCLComm&) = delete;
  CNCLComm& operator=(const CNCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  CNCLComm& operator=(CNCLComm&& other) = delete;

  // Move constructable
  CNCLComm(CNCLComm&& other) { // NOSONAR
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(cncl_comm_, other.cncl_comm_);
    std::swap(aborted_, other.aborted_);
  }

  cnclComm_t getCnclComm() {
    return cncl_comm_;
  };

  void cnclCommAbort(
      c10::optional<std::string> comm_failure_reason = c10::nullopt);

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  c10::optional<std::string> getCnclCommFailureReason() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return comm_failure_reason_;
  }

  cnclResult_t checkForCnclError() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (cncl_async_err_ != CNCL_RET_SUCCESS) {
      return cncl_async_err_;
    }
    cncl_async_err_ = cnclGetCommAsyncError(cncl_comm_);
    return cncl_async_err_;
  }

  cnclCliqueId getCnclId() {
    return cncl_id_;
  }

  friend class ProcessGroupCNCL;

 protected:
  cnclComm_t cncl_comm_;
  bool aborted_;
  mutable std::mutex mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  cnclResult_t cncl_async_err_;
  cnclCliqueId cncl_id_;
  // Optional reason for communicator failure, provided by ProcessGroupCNCL
  // for better error messaging.
  c10::optional<std::string> comm_failure_reason_;
};

// ProcessGroupCNCL implements CNCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All CNCL functions provided by this class are asynchronous functions. More
// specifically, each CNCL call is scheduled on a separate MLU stream that is
// different from the current MLU stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibilty to make sure that the MLU stream their
// code works on needs to wait for the CNCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkCNCL::wait() or WorkCNCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Note that WorkCNCL::isSuccess() and WorkCNCL::isCompleted() will always
// return true since ProcessGroupCNCL is single threaded. Every single CNCL
// or MLU failure will simply raise std::runtime_error.
//
// Therefore, WorkCNCL::exception() is not supported since isSuccess() always
// returns true.
//
// Also note that WorkCNCL::finishedMLUExecution() is a helper function only
// provided by ProcessGroupCNCL to check if the CNCL operation of WorkCNCL has
// finished execution on the MLU (not just scheduled).
//
// Example on using the CNCL process group
//
//   ProcessGroupCNCL pg(store, rank, size);
//   std::shared_ptr<WorkCNCL> work = pg.allreduce(tensors);
//
//   // At this point, CNCL kernel has already by streamd successfully
//   // Now, let current stream wait for the CNCL to finish, originally this
//   function is
//   // async operation as well, but currently MLU is sync.
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_MLU_API ProcessGroupCNCL : public c10d::Backend {
 public:
  class WorkCNCL : public c10d::Work,
                   public std::enable_shared_from_this<WorkCNCL> {
   public:
    // Constructor takes a list of MLU devices
    WorkCNCL(
        const std::vector<at::Device>& devices,
        int rank,
        c10d::OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt,
        bool desyncDebug = false); // NOLINT

    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkCNCL(const WorkCNCL& w);

    virtual ~WorkCNCL();

    // Checks if the NCCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of CNCL, it checks
    // if the CNCL operation has completed on the MLU in its own CNCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for CNCL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void abort() override;

    // Let current stream wait on the completing of the CNCL work
    // Throws on exceptions
    void synchronize() override;

    // Synchronize streams by blocking each on the CNCL stream
    void synchronizeStreams();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    std::vector<at::Tensor> result() override;

    // Helper function that returns True if the WorkCNCL object has timed out
    // and False otherwise.
    bool timedOut();

    // Helper function used in CUDA Stream callbacks to complete WorkNCCL
    // objects and throw exceptions when neeeded.
    void handleCNCLGuard(ErrorHandlingMode async_error_handling);

    // Helper function that checks if the CNCL kernels have finished
    // execution on the MLUs
    bool finishedMLUExecution();

    // Helper function that sets an exception_ptr on the WorkCNCL object.
    void setException(std::exception_ptr exception_ptr);

   protected:
    // The cached list of MLU devices to operate on
    std::vector<at::Device> devices_;

    // Clone of blocking_wait_ from ProcessGroupCNCL.
    bool blocking_wait_ = false;

    // Tensors used for barrier op
    std::vector<at::Tensor> barrier_tensors_;

    // The CNCL communicators used for this work item.
    std::vector<std::shared_ptr<CNCLComm>> cncl_comms_;

    // The start MLU events of CNCL operator tracking this work item on
    // multiple MLU devices. These start MLU events are needed by desync
    // debugging if enabled.
    std::shared_ptr<std::vector<torch_mlu::MLUEvent>> cncl_start_events_;

    // The end MLU events of CNCL operator tracking this work item on
    // multiple MLU devices.
    std::shared_ptr<std::vector<torch_mlu::MLUEvent>> cncl_end_events_;

    // Clone of op_timeout_ from ProcessGroupCNCL.
    std::chrono::milliseconds op_timeout_;

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> work_start_time_;

    // Record the collective sequential number.
    uint64_t seq_;

    // Wrapper method for the static checkForCNCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForCNCLErrors(
        const std::vector<std::shared_ptr<CNCLComm>>& cncl_comms) const;

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkCNCL& work_cncl);

   private:
    // Helper function for synchronize
    void synchronizeInternal(std::chrono::milliseconds timeout);

    // Just checks whether MLU execution has completed, without modifying
    // exception_ptr.
    bool finishedMLUExecutionInternal() const;

    // Just checks whether GPU execution has started, without modifying
    // exception_ptr.
    bool startedMLUExecutionInternal() const;

    // Checks for NCCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Checks for CNCL errors and throws an appropriate exception.
    void checkAndThrowException();

    // This function iterates through the list of WorkNCCL objects in the
    // workList_ corresponding to incomplete collectives and then aborts NCCL
    // communicators associated with timed out collectives.
    void abortTimedOutCollectives(
        std::unordered_set<std::string>& aborted_comm_ids);

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // Store a reference to CNCL collective's outputs, used by result and to
    // give a more descriptive message when representing the Work as a string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<c10d::Store> store_;

    // Indicates if the cncl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool start_trace_updated_{false};
    // Clone of avoidRecordStreams_ from ProcessGroupCNCL.
    bool avoid_record_streams_ = false;

    // TORCH_CNCL_AVOID_RECORD_STREAMS implementation helper.
    // Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStreams, just after user-facing
    // stream(s) are synced with the cncl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;

    friend class ProcessGroupCNCL;
  };

  struct Options : c10d::Backend::Options {
    // NOTE: timeout in ProcessGroupCNCL::Options denote the timeout for
    // operations. This is only used when blocking_wait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule CNCL operations on high priority MLU streams
    bool is_high_priority_stream;
  };

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }
  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any CNCL communicators. A single CNCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another CNCL
  // communicator. These CNCL communicators are cached and reused if possible.
  ProcessGroupCNCL(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  virtual ~ProcessGroupCNCL();

  const std::string getBackendName() const override {
    return std::string(CNCL_BACKEND_NAME);
  }

  // Function that runs as part of a separate thread and checks for errors on
  // CNCL communicators. We need a separate thread to check for CNCL errors
  // since we can't rely on the user calling certain methods like wait(),
  // isCompleted() etc. to detect and remediate errors. In addition to this, we
  // need a mechanism to safely abort and remove CNCL communicators from our
  // cache. This can be done cleanly by having a thread for the ProcessGroupCNCL
  // class. Attempting to modify the communicator cache from the WorkCNCL class
  // might run into issues with object lifetime since the ProcessGroupCNCL
  // object might get destroyed before the WorkCNCL object.
  void cnclCommWatchdog();

  void cnclCommWatchdogInternal();

  // This function iterates through the list of WorkCNCL objects in the
  // workList_ corresponding to incomplete collectives and then aborts CNCL
  // communicators associated with timed out collectives.
  void abortTimedOutCollectives(
      std::unordered_set<std::string>& aborted_comm_ids);

  void startCoalescing() override;

  c10::intrusive_ptr<c10d::Work> endCoalescing() override;

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::Work> _broadcast_oop(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions());

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_oop(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions());

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dst_rank,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int src_rank,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  // Create a new ProcessGroupCNCL instance
  static c10::intrusive_ptr<c10d::Backend> createProcessGroupCNCL(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout);

  static void groupStart();

  static void groupEnd();

  void workCleanupLoop();

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;
  void setSequenceNumberForGroup() override;

  // get a cnclComm_t.
  int64_t getCnclComm(int rankid);

 protected:
  static const int64_t k_watchdog_thread_sleep_millis;
  static const int64_t k_work_cleanup_thread_sleep_millis;

  // The store is used to broadcast the CNCL unique ID of rank 0.
  c10::intrusive_ptr<c10d::Store> store_;

  bool store_error_{false};

  // The store keys to trace the last CNCL collective kernel MLU events - start
  // event and end event respectively. These are used to do desync root cause
  // analysis.
  const std::string trace_key_start_;
  const std::string trace_key_end_;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode async_error_handling_ = NoHandling;

  // Whether or not to enable timeout root cause analysis.
  bool desync_debug_;

  // Watchdog thread which looks for errors on the cached CNCL communicators.
  std::thread cncl_comm_watchdog_thread_;

  // Condition variable to control how long the watchdog thread waits.
  std::condition_variable watchdog_cv_;

  // Mutex for watchdog.
  std::mutex watchdog_cv_mutex_;

  // Watchdog thread which looks for errors on the cached CNCL communicators.
  std::thread cncl_comm_watchdog_thread;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminate_process_group_;

  // Map from cnclCliqueId to appropriate communicator.
  std::unordered_map<std::string, std::vector<std::shared_ptr<CNCLComm>>>
      cncl_id_to_comm_map_;

  // Mutex to Guard workMetaList_
  std::mutex work_meta_list_mutex_;

  // Condition Variable for timeout thread sleep
  std::condition_variable work_meta_list_cv_;

  // Vector to Store WorkNCCL pointers
  std::list<ProcessGroupCNCL::WorkCNCL> work_meta_list_;

  // Thread that removes CNCL Work upon timeout
  std::thread work_cleanup_thread_;

  // Set of communicators that this process group has aborted and their
  // cnclCliqueId has been written to the store. We don't need a lock
  // for this map since only the watchdog thread accesses this set. The
  // set contains the string representation of cnclCliqueId.
  std::unordered_set<std::string> aborted_comms_;

  // Add Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>);

  // Helper that broadcasts cncl clique ID to all ranks through the store
  void broadcastCNCLCliqueID(
      cnclCliqueId* cncl_id,
      const bool is_p2p_op,
      const std::string& p2p_key,
      const int p2p_rank);

  // Helper that either looks up the cached CNCL communicators or creates
  // a new set of CNCL communicators as a cache entry
  std::vector<std::shared_ptr<CNCLComm>>& getCNCLComm(
      const std::string& devices_key,
      const std::vector<at::Device>& devices,
      c10d::OpType op_type,
      const int p2p_rank = 0,
      const bool is_send_recv_self = false);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForCNCLErrors(
      const std::vector<std::shared_ptr<CNCLComm>>& cncl_comms);

  virtual c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> initWork(
      std::vector<at::Device> devices,
      int rank,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr,
      const c10::optional<std::vector<at::Tensor>>& inputs = c10::nullopt);

  const c10::intrusive_ptr<Options> options_;

  // The number of CNCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t cncl_comm_counter_{0};

  // The CNCL communicator that the process group has cached.
  // The key is a list of MLU devices that an operation is operating on
  // The MLU devices are stored in a device sequence and the cache CNCL
  // communicator is associated with this MLU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  std::unordered_map<std::string, std::vector<std::shared_ptr<CNCLComm>>>
      dev_cncl_comm_map_;

  // The MLU streams used by CNCL kernels
  std::unordered_map<std::string, std::vector<torch_mlu::MLUStream>>
      cncl_streams_;

  // The MLU events used to sync CNCL streams
  std::unordered_map<std::string, std::vector<torch_mlu::MLUEvent>>
      cncl_events_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Whether or not TORCH_CNCL_AVOID_RECORD_STREAMS was set
  bool avoid_record_streams_ = false;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blocking_wait_ = false;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  bool coalescing_active_ = false;

  // Stores device indexes for all collectives run inside a coalescing block
  std::vector<std::vector<at::Device>> coalescedDevices_;

 private:
  // Helper that encapsulates work shared across all collective communication
  // primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collective(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communicaiton primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> pointToPoint(
      std::vector<at::Tensor>& tensors,
      Fn fn,
      int peer,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr);

  // Checks for CNCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForCNCLErrorsInternal(
      const std::vector<std::shared_ptr<CNCLComm>>& cncl_comms);

  // The number of active cnclGroupStart() calls. This counter will be increased
  // by 1 when cnclGroupStart() is called and decreased by 1 when cnclGroupEnd()
  // is called.
  static thread_local uint64_t cnclActiveGroupCounter_;

  // Counting for the sequential number of CNCL collective call.
  uint64_t seq_{0};

  // Mutex to guard maps like dev_cncl_comm_map_
  std::mutex mutex_;
};

} // namespace torch_mlu
