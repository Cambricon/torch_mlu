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

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <torch/all.h>
#include <unordered_map>
#include <future>

#include "cncl.h"
#include "framework/core/MLUEvent.h"
#include "aten/utils/tensor_util.h"

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

namespace torch_mlu {

// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
static std::vector<std::string> TORCH_CNCL_BLOCKING_WAIT = {
    "TORCH_CNCL_BLOCKING_WAIT",
    "TORCH_NCCL_BLOCKING_WAIT",
    "NCCL_BLOCKING_WAIT"};

// Environment variable which controls whether or not we perform Async Error
// Handling with CNCL.
static std::vector<std::string> TORCH_CNCL_ASYNC_ERROR_HANDLING = {
    "TORCH_CNCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING"};

// Environment Variable to control whether Desync Debug is enabled.
// This variable must be set together with CNCL_ASYNC_ERROR_HANDLING.
static std::vector<std::string> TORCH_CNCL_DESYNC_DEBUG = {
    "TORCH_CNCL_DESYNC_DEBUG",
    "TORCH_NCCL_DESYNC_DEBUG",
    "NCCL_DESYNC_DEBUG"};

// Enable monitoring thread which aborts the process when the ProcessGroupCNCL
// Watchdog thread gets stuck and no heartbeat is detected after
// TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling MLU/CNCL
// APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged
// time than necessary tying up cluster resources.
static std::vector<std::string> TORCH_CNCL_ENABLE_MONITORING = {
    "TORCH_CNCL_ENABLE_MONITORING",
    "TORCH_NCCL_ENABLE_MONITORING"};

// Control whether dumping debug info on watchdog
// timeout is enabled. This variable must be set together with
// TORCH_CNCL_ENABLE_MONITORING=1 and TORCH_CNCL_TRACE_BUFFER_SIZE > 0.
static std::vector<std::string> TORCH_CNCL_DUMP_ON_TIMEOUT = {
    "TORCH_CNCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_DUMP_ON_TIMEOUT"};

// Enable recording start-events for all ProcessGroupCNCL collectives, and
// compute accurate collective timing per-collective. (Note: end-events are
// recorded by default. Turn on this flag can increase chances of a watchdog
// hang due to performing a MLU event query which eventually calls
// cudaEventElapsedTime() API.
static std::vector<std::string> TORCH_CNCL_ENABLE_TIMING = {
    "TORCH_CNCL_ENABLE_TIMING",
    "TORCH_NCCL_ENABLE_TIMING",
    "NCCL_ENABLE_TIMING"};

// Control the watchdog heartbeat timeout period after which the monitoring
// thread will abort the process.
static std::vector<std::string> TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"};

// Control the interval inside the watchdog thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_CNCL_COORD_CHECK_MILSEC = {
    "TORCH_CNCL_COORD_CHECK_MILSEC",
    "TORCH_NCCL_COORD_CHECK_MILSEC"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_CNCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_CNCL_WAIT_TIMEOUT_DUMP_MILSEC",
    "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// The maximum number of events we store in the flight recorder's ring buffer.
// (One event could be the start or end of a collective, for example).
static std::vector<std::string> TORCH_CNCL_TRACE_BUFFER_SIZE = {
    "TORCH_CNCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_TRACE_BUFFER_SIZE"};

constexpr const char* CNCL_BACKEND_NAME = "cncl";

constexpr const char* TIMEOUT_DUMP = "timeout_dump";

constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds

constexpr auto kProcessGroupCNCLDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

// NoHandling: do not handle asynchronous CNCL errors
// TearDown: tear down process upon error, see `WorkCNCL::handleException`
// CleanUpOnly: just clean up collectives and abort communicators without
// tearing down process SkipCleanUp: (this is a temporary option and can be
// removed in future) tear down process without cleaning up CNCL communicators.
// This should be used as a last resort in case `cnclCommAbort` itself is
// hanging
enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

// If set, ProcessGroupCNCL doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// Instead, it stashes live references to those tensors until after
// user-facing streams are synced with comm streams.
// See stashed_for_allocator_safety_ below.
static std::vector<std::string> TORCH_CNCL_AVOID_RECORD_STREAMS = {
    "TORCH_CNCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_AVOID_RECORD_STREAMS"};

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
    std::unique_lock<std::mutex> lock(mutex_);
    if (aborted_) {
      auto commFailureMsg = comm_failure_reason_ != c10::nullopt
          ? c10::str(
                " Original reason for failure was: ", *comm_failure_reason_)
          : "";
      TORCH_CHECK_WITH(
          DistBackendError,
          false,
          c10::str(
              "CNCL communicator was aborted on rank ",
              rank_,
              ". ",
              commFailureMsg));
    }
    return cncl_comm_;
  }

  void cnclCommAbort(
      std::optional<std::string> comm_failure_reason = c10::nullopt);

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  std::optional<std::string> getCnclCommFailureReason() const {
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
  std::optional<std::string> comm_failure_reason_;
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
        at::Device& device,
        int rank,
        c10d::OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = c10::nullopt,
        bool desyncDebug = false,
        bool enableTiming = false,
        c10d::DebugLevel distDebugLevel = c10d::DebugLevel::Off);

    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkCNCL(const WorkCNCL& w);

    virtual ~WorkCNCL();

    // Checks if the CNCL kernel has started to execute.
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
    // Throws on exceptions. Blocking operation, which will wait for work
    // completion.
    void synchronize() override;

    // Synchronize stream by blocking each on the CNCL stream
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the CNCL kernels have finished
    // execution on the MLUs
    bool finishedMLUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkCNCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkCNCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkCNCL object.
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = c10::nullopt);

    std::vector<at::Tensor> result() override;

   protected:
    // The cached list of MLU devices to operate on
    at::Device device_;

    // The start MLU events of CNCL operator tracking this work item on
    // multiple MLU devices. These start MLU events are needed by desync
    // debugging if enabled.
    std::shared_ptr<torch_mlu::MLUEvent> cncl_start_event_;

    // The end MLU events of CNCL operator tracking this work item on
    // multiple MLU devices.
    std::shared_ptr<torch_mlu::MLUEvent> cncl_end_event_;

    // The CNCL communicators used for this work item.
    std::shared_ptr<CNCLComm> cnclComm_;

    // Tensors used for barrier op
    at::Tensor barrierTensor_;

    // Clone of blockingWait_ from ProcessGroupCNCL.
    bool blockingWait_ = false;

    // Clone of avoidRecordStreams_ from ProcessGroupCNCL.
    bool avoidRecordStreams_ = false;

    // Clone of opTimeout_ from ProcessGroupCNCL.
    std::chrono::milliseconds opTimeout_;

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number.
    uint64_t seq_;

    // Indicates if the cncl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

    // Wrapper method for the static checkForCNCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForCNCLErrors();

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkCNCL& workCNCL);

   private:
    // Helper function for synchronize
    void synchronizeInternal(std::chrono::milliseconds timeout);

    // Checks for CNCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Just checks whether MLU execution has started, without modifying
    // exception_ptr.
    bool startedMLUExecutionInternal() const;

    // Just checks whether MLU execution has completed, without modifying
    // exception_ptr.
    bool finishedMLUExecutionInternal() const;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<c10d::Store> store_;

    // c10d::Store a reference to CNCL collective's outputs, used by result and
    // to give a more descriptive message when representing the Work as a
    // string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_CNCL_AVOID_RECORD_STREAMS implementation helper.
    // c10d::Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the cncl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    bool timingEnabled_;
    // unique id used to tell the trace buffer that this
    // work has completed
    std::optional<uint64_t> trace_id_;
    c10d::DebugLevel distDebugLevel_;

    friend class ProcessGroupCNCL;
  };

  struct Options : c10d::Backend::Options {
    // NOTE: timeout in ProcessGroupCNCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule CNCL operations on high priority MLU streams
    bool is_high_priority_stream;

    std::vector<uint64_t> global_ranks_in_group;
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

  uint64_t getUid() {
    return static_cast<uint64_t>(uid_);
  }

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

  // Watchdog's inside loop.
  // Takes care of cleaning up completed work, and aborting upon failure or
  // timeout.
  void watchdogHandler();

  // Desync debug helper
  void logWorkStart(WorkCNCL& work);

  // Desync debug helper
  void logWorkEnd(WorkCNCL& work);

  // This function iterates through the list of WorkCNCL objects in the
  // workList_ corresponding to incomplete collectives and then aborts CNCL
  // communicators associated with timed out collectives.
  void abortTimedOutCollectives(
      std::unordered_set<std::string>& aborted_comm_ids);

  void startCoalescing() override;

  c10::intrusive_ptr<c10d::Work> endCoalescing() override;

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<c10d::Work> endCoalescing(c10d::OpType optype);

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::Work> _broadcast_oop(
      at::Tensor& output_tensors,
      at::Tensor& input_tensors,
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
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
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

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  // Helper function for iteratively aborting communicators in the provided map
  void abortCommsFromMap(
      std::unordered_map<std::string, std::shared_ptr<CNCLComm>>& cnclComms_map,
      std::optional<std::string> abortReason);

  // Provides an API to abort the ProcessGroup (similar to cnclCommAbort)
  // instead of relying on ProcessGroupCNCL destructor.
  // return true if abort is successful, otherwise false
  bool abort(std::optional<std::string> abortReason = c10::nullopt);

  void shutdown();

  // Returns the global rank of the device. This function assumes that users
  // always create a default global process group(PG) which includes all
  // devices. It is called in the constructor of ProcessGroupCNCL, so it always
  // return the rank_ of the the very first PG created, aka, default global PG.
  const int& globalRank() const;

  // Returns the global ranks of a PG.
  const std::vector<uint64_t>& groupRanks() const;

  // get a cnclComm_t.
  int64_t getCnclComm(int rankid);

 protected:
  // Function that runs as part of a separate thread aside from watchdog
  // thread because we need to check the heartbeat from watchdog thread
  // so that when we get stuck in some CNCL/MLU calls,
  // we can dump the debugging information and abort the process.
  virtual void heartbeatMonitor();

  // Function that directly trigger std::abort so that the whole process
  // gets terminated.
  virtual void terminateProcess(std::string errMsg);

  static const int64_t k_watchdog_thread_sleep_millis;
  static const int64_t k_work_cleanup_thread_sleep_millis;

  // A helper function to wait for a future to complete or timeout.
  void waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      bool throwException = false);

  // When watchdog timeout, this function will be called and return debug info
  // for users. For now we only get information from retrieveDesyncReport.
  // We are working on enabling more useful debug information for watchdog
  // timeout.
  virtual std::string getCNCLWatchdogDebugInfo();

  // The store is used to broadcast the CNCL unique ID of rank 0. This store
  // comes with prefix and it is different across ProcessGroup CNCL instances
  // (aka, different ProcessGroups).
  c10::intrusive_ptr<c10d::Store> store_;

  // Reference to the store without prefix so that keys are same across all
  // ProcessGroup CNCL instances and (key, value) pairs written to the store are
  // global.
  c10::intrusive_ptr<c10d::Store> globalStore_;

  bool storeError_{false};

  // The store keys to trace the last CNCL collective kernel MLU events - start
  // event and end event respectively. These are used to do desync root cause
  // analysis.
  const std::string traceKeyStart_;
  const std::string traceKeyEnd_;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode async_error_handling_ = NoHandling;

  // Whether or not to enable timeout root cause analysis.
  bool desync_debug_;

  // Whether or not to dump debug info on timeout
  bool dumpOnTimeout_;

  // Whether or not to create start MLUEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desync_debug_
  // is set to true.
  std::atomic<bool> enableTiming_;

  // Watchdog thread which looks for errors on the cached CNCL communicators.
  std::thread cncl_comm_watchdog_thread_;

  // Counting for the sequential number of CNCL collective call.
  uint64_t seq_{0};

  // Incrementing counter for logical operations (collective or p2p) issued on
  // the ProcessGroup
  uint64_t op_id_{0};

  // the sequential number of the last colletive enstreamd into workMetaList_
  // This is useful for indentifying a rank that has not join a collective
  uint64_t lastEnstreamdSeq_;

  // the sequential number of the last colletive completed marked by
  // the watchdog thread
  uint64_t lastCompletedSeq_;
  // Mutex for watchdog.
  std::mutex watchdog_cv_mutex_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminate_process_group_;

  // Map from cnclCliqueId to appropriate communicator.
  std::unordered_map<std::string, std::shared_ptr<CNCLComm>>
      cncl_id_to_comm_map_;

  // Whether or not we should terminate the heartbeat monitoring threads.
  std::atomic<bool> terminateHeartbeatMonitorThread_;

  // The time interval used for deciding whether there is no watchdog heartbeat.
  int heartbeatTimeoutInSec_;

  // timeout for the dump to finish.
  int waitTimeoutDumpInMilSec_;

  // We gate the heartbeat monitor thread so that we can roll it out gradually.
  std::atomic<bool> monitorThreadEnabled_;

  // Monitor thread which checks the heartbeat of Watchdog thread.
  // If the monitor thread finds there is no heartbeat, it will dump debug info
  // and then kill the watchdog thread to avoid hang.
  std::thread cnclHeartbeatMonitorThread_;

  // Interval of check coordinated signals in ProcessGroupCNCL from other ranks
  // e.g., trigger the dump of the debugging info for timeout when notified.
  int coordCheckIntervalMilSec_;

  // Size of ring buffer where we store CNCL Traces for debugging.
  int cnclTraceBufferSize_;

  // Whether we are in the shutdown mode when we are trying to get debug info,
  // such as desync report.
  std::atomic<bool> collectiveDebugInfoMode_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessiable from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the 3 conditions: 1: this flag is set to true by the watchdog thread when
  // it detects a timeout. 2: timeout signal is received from
  // other ranks through tcpstore 3: no heartbeat of watchdog Note that only the
  // monitor thread from PG0 should dump the debug info and only once
  static std::atomic<bool> shouldDump_;

  // Mutex to Guard monitorWakeUpCV_
  std::mutex monitorMutex_;

  // Condition Variable for monitor thread to wake up early
  std::condition_variable monitorWakeUpCV_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaList_mutex_;

  // Condition Variable for timeout thread sleep
  std::condition_variable workMetaListCV_;

  // Vector to c10d::Store WorkCNCL pointers
  std::list<ProcessGroupCNCL::WorkCNCL> workMetaList_;

  std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;

  // Thread that removes CNCL Work upon timeout
  std::thread work_cleanup_thread_;

  std::string logPrefix_;

  size_t uid_;

  // Number of devices on this node.
  int localDeviceCount_{0};

  // Set of communicators that this process group has aborted and their
  // cnclCliqueId has been written to the store. We don't need a lock
  // for this map since only the watchdog thread accesses this set. The
  // set contains the string representation of cnclCliqueId.
  std::unordered_set<std::string> aborted_comms_;

  // Add Work Pointer to workVector
  void workEnstream(c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>);

  // Helper that broadcasts cncl clique ID to all ranks through the store
  void broadcastCNCLCliqueID(
      cnclCliqueId* cncl_id,
      const bool is_p2p_op,
      const std::string& p2p_key,
      const int p2p_rank);

  // Helper that either looks up the cached CNCL communicators or creates
  // a new set of CNCL communicators as a cache entry
  std::shared_ptr<CNCLComm> getCNCLComm(
      const std::string& device_key,
      at::Device& device,
      c10d::OpType op_type,
      const int p2p_rank = 0,
      const bool is_send_recv_self = false);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForCNCLErrors(
      std::shared_ptr<CNCLComm>& cncl_comms);

  // Ensure thaht if record is True, the work obj will be enstreamd via
  // workEnstream
  virtual c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> initWork(
      at::Device device,
      int rank,
      c10d::OpType opType,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false);

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
  std::unordered_map<std::string, std::shared_ptr<CNCLComm>> dev_cncl_comm_map_;

  // The MLU streams used by CNCL kernels
  std::unordered_map<std::string, torch_mlu::MLUStream> cncl_streams_;

  // The MLUEvents used to sync CNCL streams
  std::unordered_map<std::string, torch_mlu::MLUEvent> cncl_events_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Whether or not TORCH_CNCL_AVOID_RECORD_STREAMS was set
  bool avoidRecordStreams_ = false;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // c10d::Stores device indexe for all collectives run inside a coalescing
  // block
  at::Device coalescedDevice_ = at::Device("mlu");

  // c10d::Stores communicator for all collectives run inside a coalescing block
  std::shared_ptr<CNCLComm> coalescedComm_;

  // In the timeout case and we will dump debug info such as the CNCL flight
  // recorder to storage. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  bool dumpDebuggingInfo();

 private:
  // Helper that encapsulates work shared across all collective communication
  // primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::Work> collective(
      at::Tensor& inputs,
      at::Tensor& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr,
      bool avoidRecordStreams = false);

  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collectiveCoalesced(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr,
      bool avoid_record_streams = false);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communicaiton primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> pointToPoint(
      at::Tensor& tensors,
      Fn fn,
      int peer,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr);

  c10::intrusive_ptr<c10d::Work> allreduce_impl(
      at::Tensor& tensor,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions());

  // Checks for CNCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForCNCLErrorsInternal(
      std::shared_ptr<CNCLComm>& cncl_comms);

  // Generates a prefix that is unique to this process group and rank, for
  // disambiguating logs
  std::string createLogPrefix() const;

  // Returns the unique prefix created in createLogPrefix
  const std::string& logPrefix() const;

  // The number of active cnclGroupStart() calls. This counter will be increased
  // by 1 when cnclGroupStart() is called and decreased by 1 when cnclGroupEnd()
  // is called.
  static thread_local uint64_t cnclActiveGroupCounter_;

  std::exception_ptr watchDogException_ = nullptr;

  // Mutex to guard maps like dev_cncl_comm_map_
  std::mutex mutex_;

  // Heartbeat of watchdog thread.
  std::atomic_uint64_t heartbeat_;
};

TORCH_MLU_API std::string dump_cncl_trace();

// Gets a mutable reference to a global optional function.  Heartbeat Monitor
// will query this function and if available, call it to dump traces. Inside
// fbcode, we store a function here that uses an internal tool for process
// tracing
TORCH_MLU_API std::optional<std::function<std::string()>>&
get_cpp_trace_dumper();

// Similar to get_cpp_trace_dumper, this stores a function defined in
// torch-python layer that lets us check whether the GIL can be acquired,
// helpful for instrumenting in cases where a hang was observed.
typedef bool (*gil_checker_t)();

TORCH_MLU_API gil_checker_t& get_gil_checker();

} // namespace torch_mlu
