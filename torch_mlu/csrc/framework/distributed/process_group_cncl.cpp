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

#include "process_group_cncl.hpp"
#include "framework/graphs/MLUGraph.h"
#include "framework/graphs/MLUGraphUtils.h"
#include "TraceUtils.h"

#include <map>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include "cncl_utils.h"
#include "framework/core/stream_guard.h"
#include "aten/utils/utils.h"

#ifdef TEST_COVERAGE
extern "C" void __gcov_flush();
#endif

namespace torch_mlu {

constexpr const char* const k_cncl_aborted_comm_store_key = "CNCLABORTEDCOMM";

#if defined(__linux__)
struct DumpPipe {
  DumpPipe(int rank) {
    std::string fileStem =
        getCvarString({"TORCH_CNCL_DEBUG_INFO_PIPE_FILE"}, "");
    if (fileStem.empty() ||
        getCvarInt({"TORCH_CNCL_TRACE_BUFFER_SIZE"}, 0) <= 0) {
      return;
    }
    TORCH_CHECK(!fileStem.empty(), "TORCH_CNCL_DEBUG_INFO_TEMP_FILE is empty");
    std::string filename = c10::str(fileStem, rank, ".pipe");
    TORCH_CHECK(
        unlink(filename.c_str()) != -1 || errno == ENOENT,
        "Error removing existing named pipe ",
        filename);
    TORCH_CHECK(
        mkfifo(filename.c_str(), 0666) != -1,
        "Error creating named pipe ",
        filename);
    fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
    LOG(INFO) << "Pipe file " << filename
              << " has been opened, write to it to trigger CNCL Debug Dump.";
    TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
  }
  bool shouldDump() {
    if (fd_ == -1) {
      return false;
    }
    char buf[128];
    // non-blocking from O_NONBLOCK above.
    // Ignore EINTR because we already will poll this
    // again later.
    ssize_t bytesRead = read(fd_, &buf, 128);
    return bytesRead > 0;
  }
  ~DumpPipe() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

 private:
  int fd_ = -1;
};
#else
struct DumpPipe {
  DumpPipe(int rank) {}
  bool shouldDump() {
    return false;
  }
};
#endif

CNCLComm::~CNCLComm() noexcept {
  // Add lock in this destructor, as aborted_ needs to be read after memory
  // barrier here.
  std::unique_lock<std::mutex> lock(mutex_);
  if (cncl_comm_ && !aborted_) {
    // TODO(zhiguangda): use cnclCommAbort when torch_mlu support
    // environment variable like ENABLE_NCCL_ERROR_CHECKING
    C10D_CNCL_ASSERT(cnclDestroyComms(&cncl_comm_, 1));
  }
}

std::shared_ptr<CNCLComm> CNCLComm::create(
    int numRanks,
    int rank,
    int device,
    const cnclCliqueId_t clique_id) {
  auto comm = std::make_shared<CNCLComm>();
  C10D_CNCL_CHECK(
      cnclInitComms(
          &(comm->cncl_comm_), 1, &device, &rank, numRanks, clique_id),
      c10::nullopt);
  comm->rank_ = rank;
  comm->cncl_id_ = *clique_id;
  return comm;
}

void CNCLComm::cnclCommAbort(std::optional<std::string> comm_failure_reason) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (aborted_) {
    // Should not abort twice.
    return;
  }

  // Set true failure reason if provided by ProcessGroupCNCL (e.g. work
  // timeout)
  comm_failure_reason_ = comm_failure_reason;

  C10D_CNCL_CHECK(cnclAbortComm(cncl_comm_), comm_failure_reason_);
  aborted_ = true;
  cncl_comm_ = nullptr;

  if (cncl_async_err_ == CNCL_RET_SUCCESS) {
    cncl_async_err_ = CNCL_RET_ERR_SYSTEM;
  }

  // Clear the stream in cncl_streams that are associated with cncl_comm_.
  clearCnclStream(&cncl_id_);
}

namespace {

// RAII helper class to manage CNCL group API and MLU free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct AutoCnclGroup {
  AutoCnclGroup() {
    // TODO(zhiguangda): using lock in order to follow MLU,
    // but whether MLU needs to be locked is controversial
    (torch_mlu::getFreeMutex())->lock();
    C10D_CNCL_CHECK(cnclGroupStart(), c10::nullopt);
  }
  ~AutoCnclGroup() noexcept(false) {
    C10D_CNCL_CHECK(cnclGroupEnd(), c10::nullopt); // NOSONAR
    (torch_mlu::getFreeMutex())->unlock();
  }
};

// Returns exception's what() given an exception_ptr instance.
std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
  TORCH_CHECK(exceptionPtr != nullptr);
  try {
    std::rethrow_exception(exceptionPtr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}

// CNCL op mapping
const std::map<c10d::ReduceOp, cnclReduceOp_t> cncl_op = {
    {c10d::ReduceOp::MIN, cnclMin},
    {c10d::ReduceOp::MAX, cnclMax},
    {c10d::ReduceOp::SUM, cnclSum},
    {c10d::ReduceOp::PRODUCT, cnclProd},
};

// CNCL type typing
std::map<at::ScalarType, cnclDataType_t> cncl_data_type = {
    {at::kChar, cnclInt8},
    {at::kByte, cnclUint8},
    {at::kFloat, cnclFloat},
    {at::kInt, cnclInt32},
    {at::kLong, cnclInt64},
    {at::kHalf, cnclHalf},
    {at::kDouble, cnclFloat},
    {at::kBool, cnclUint8},
    {at::kBFloat16, cnclBfloat16}};

// Helper function that gets the data type and issues error if not supported
cnclDataType_t getCnclDataType(at::ScalarType type) {
  try {
    return cncl_data_type.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for CNCL process group");
  }
}

cnclReduceOp_t getCnclReduceOp(
    const c10d::ReduceOp reduce_op,
    at::Tensor& input) {
  try {
    if (reduce_op == c10d::ReduceOp::SUM && input.scalar_type() == at::kBool) {
      // For bool tensors, map sum to max, which both represent a bitwise or.
      // This is to prevent overflow issues with sum, since we use uint8 to
      // represent a bool (see cnclDataType mapping).
      return cnclMax;
    }
    return cncl_op.at(reduce_op);
  } catch (const std::out_of_range& e) {
    switch (reduce_op) {
      case c10d::ReduceOp::AVG:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.AVG with CNCL");
        break;
      case c10d::ReduceOp::BAND:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with CNCL");
        break;
      case c10d::ReduceOp::BOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with CNCL");
        break;
      case c10d::ReduceOp::BXOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with CNCL");
        break;
      default:
        C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
        break;
    }
  }
}

// Get a key string from device
std::string getKeyFromDevice(at::Device& device) {
  return std::to_string(device.index());
}

std::string getKeySendRecv(int my_rank, int peer) {
  int low_rank = my_rank < peer ? my_rank : peer;
  int high_rank = my_rank < peer ? peer : my_rank;
  std::string send_recv_pair =
      std::to_string(low_rank) + ":" + std::to_string(high_rank);
  return send_recv_pair;
}

// Get device from tensor
at::Device getDevice(at::Tensor& tensor) {
  return tensor.device();
}

// [Sync Queue] Helper that lets the input cncl_stream to wait for the current
// stream. CNCL communications run on cncl_stream, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// cncl_stream cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on cncl_stream finish. This
// can be achieved by calling torch_mlu::MLUCachingAllocator::recordStream,
// which remembers the usage stream (cncl_stream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
void syncStream(
    at::Device device,
    torch_mlu::MLUEvent& cncl_event,
    torch_mlu::MLUStream& cncl_stream) {
  cncl_event.place(torch_mlu::getCurrentMLUStream(device.index()));
  cncl_event.wait(cncl_stream);
}

// Given a cnclClistreamId, convert it to a string representation that can be
// put in the store.
std::string buildCnclUniqueIdStr(const cnclCliqueId& cncl_id) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&(cncl_id.data));
  std::ostringstream oss;
  for (const auto i : c10::irange(CNCL_CLIQUE_ID_BYTES_SIZE)) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

std::string getCnclAbortedCommStoreKey(const std::string cncl_id_str) {
  return std::string(k_cncl_aborted_comm_store_key) + ":" + cncl_id_str;
}

// Check if all tensors in a vector are arranged in continuous memory
bool checkTensorsNeedFlatten(std::vector<at::Tensor>& inputs) {
  if (inputs.empty()) {
    TORCH_CHECK(false, "Received an empty list");
  }
  auto input_base_ptr = mlu_data_ptr(torch_mlu::getMluTensorImpl(inputs[0]));
  for (int i = 0; i < inputs.size(); i++) {
    if (!inputs[i].device().is_privateuseone() || inputs[i].is_sparse()) {
      return false;
    }
    // Check tensors contiguous
    if (!inputs[i].is_contiguous(inputs[i].suggest_memory_format())) {
      return false;
    }
    auto input_impl = torch_mlu::getMluTensorImpl(inputs[i]);
    auto input_ptr = mlu_data_ptr(input_impl);
    // if not continuous return false directly
    if (input_ptr != input_base_ptr)
      return false;
    input_base_ptr = static_cast<void*>(
        static_cast<char*>(input_ptr) +
        getCnnlTypeSize(getCnnlType(input_impl)) * input_impl->numel());
  }
  return true;
}

} // namespace

const int64_t ProcessGroupCNCL::k_watchdog_thread_sleep_millis = 100;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupCNCL::cnclActiveGroupCounter_ = 0;

std::atomic<bool> ProcessGroupCNCL::shouldDump_(false);

at::Device getDeviceForRank(int rankid) {
  TORCH_CHECK(rankid >= 0, "Invalid rank id ", rankid);
  auto mluNum = torch_mlu::device_count();
  TORCH_CHECK(mluNum > 0, "Invalid MLU number ", mluNum);
  int16_t deviceIdx = static_cast<int16_t>(rankid % mluNum);
  return at::Device(c10::DeviceType::PrivateUse1, deviceIdx);
}

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupCNCL::WorkCNCL& work_cncl) {
  std::string workInfo;
  if (work_cncl.outputs_) {
    workInfo = c10::str(
        "WorkCNCL(",
        "SeqNum=",
        work_cncl.seq_,
        ", c10d::OpType=",
        opTypeToString(work_cncl.opType_),
        ", TensorShape=",
        (*work_cncl.outputs_)[0].sizes(),
        ", Timeout(ms)=",
        work_cncl.opTimeout_.count(),
        ")");
  } else {
    workInfo = c10::str(
        "WorkCNCL(",
        "SeqNum=",
        work_cncl.seq_,
        ", c10d::OpType=",
        opTypeToString(work_cncl.opType_),
        ", Timeout(ms)=",
        work_cncl.opTimeout_.count(),
        ")");
  }
  return output << workInfo;
}

ProcessGroupCNCL::WorkCNCL::WorkCNCL(
    at::Device& device,
    int rank,
    c10d::OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool desyncDebug,
    bool enableTiming,
    c10d::DebugLevel distDebugLevel)
    : Work(rank, opType, profilingTitle, inputs),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      timingEnabled_(enableTiming),
      distDebugLevel_(distDebugLevel) {
  // Creates the MLU event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = CNRT_NOTIFIER_DISABLE_TIMING_ALL.
  if (enableTiming) {
    cncl_start_event_ =
        std::make_shared<torch_mlu::MLUEvent>(CNRT_NOTIFIER_DEFAULT);
  }
  cncl_end_event_ = std::make_shared<torch_mlu::MLUEvent>(
      enableTiming ? CNRT_NOTIFIER_DEFAULT : CNRT_NOTIFIER_DISABLE_TIMING_ALL);
}

ProcessGroupCNCL::WorkCNCL::~WorkCNCL() = default;

bool ProcessGroupCNCL::WorkCNCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedMLUExecutionInternal();
}

bool ProcessGroupCNCL::WorkCNCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "WorkCNCL::isSuccess() is deprecated");
}

bool ProcessGroupCNCL::WorkCNCL::finishedMLUExecutionInternal() const {
  // Checking the work's corresponding MLU event's status
  try {
    // Checking the work's corresponding MLU events' status
    if (!cncl_end_event_->query()) {
      return false;
    }
  } catch (const std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") ==
        std::string::npos) {
      throw;
    }
    LOG(INFO) << "[Rank " << rank_
              << "] Event query failed with exception: " << e.what();
  }
  return true;
}

std::vector<at::Tensor> ProcessGroupCNCL::WorkCNCL::result() {
  return *outputs_;
}

uint64_t ProcessGroupCNCL::WorkCNCL::getSequencenumber() const {
  return seq_;
}

float ProcessGroupCNCL::WorkCNCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(
      cncl_start_event_,
      "getDuration only works if cncl_start_event_ is populated, true if timing enabled");
  TORCH_CHECK(
      cncl_end_event_,
      "getDuration only works if cncl_end_event_ is populated, which should always be true");
  return cncl_start_event_->elapsed_time(*cncl_end_event_);
}

void ProcessGroupCNCL::WorkCNCL::synchronizeStream() {
  auto current_stream = torch_mlu::getCurrentMLUStream(device_.index());
  // Block the current stream on the CNCL stream
  cncl_end_event_->wait(current_stream);

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

void ProcessGroupCNCL::WorkCNCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForCNCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(INFO) << "[Rank " << rank_ << "]"
              << " found async exception when checking for CNCL errors: "
              << getExceptionMsgFromExceptionPtr(exception_);
  }
}

// Waiting on the work's corresponding CNRT events
void ProcessGroupCNCL::WorkCNCL::synchronize() {
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupCNCL::WorkCNCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  synchronizeStream();

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? c10::nullopt : c10::make_optional(timeout));
      // Explicitly abort cnclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting cncl communicators
      // here, it was observed that MLU MLU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(),
            "Work ",
            (*this),
            " timed out in blocking wait (TORCH_CNCL_BLOCKING_WAIT=1).");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    // exception() includes timeout and error during blocking wait
    if (exception()) {
      // Abort CNCL communicators
      abort();
      // Throw exception (from main thread here)
      handleException(TearDown);
    }
  }

  // Device synchronize only after we've completed timeout checks.
  if (barrierTensor_.defined()) {
    // If we use the work to do barrier, we should block here
    auto currentStream = torch_mlu::getCurrentMLUStream(device_.index());
    TORCH_CNRT_CHECK(cnrtQueueSync(currentStream));
  }
}

// Helper that checks if the CNCL kernels are completed on the MLUs
bool ProcessGroupCNCL::WorkCNCL::finishedMLUExecution() {
  checkAndSetException();
  return finishedMLUExecutionInternal();
}

bool ProcessGroupCNCL::WorkCNCL::startedMLUExecutionInternal() const {
  // Checking the work's corresponding MLU events' status
  if (!cncl_start_event_->query()) {
    return false;
  }
  return true;
}

const std::string& ProcessGroupCNCL::WorkCNCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

bool ProcessGroupCNCL::WorkCNCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout)
    return false;

  // Timed out

  // There is already an error, we don't override it
  if (exception())
    return true;

  std::string exceptionMsg = c10::str(
      logPrefix(),
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;
  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  setException(exception_ptr);
  return true;
}

void ProcessGroupCNCL::WorkCNCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some CNCL operations have failed or timed out. Due to the ",
        "asynchronous nature of MLU kernels, subsequent MLU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupCNCL.WorkCNCL.handleException");

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

// Same as calling synchronize().
bool ProcessGroupCNCL::WorkCNCL::wait(std::chrono::milliseconds timeout) {
  synchronizeInternal(timeout);
  return true;
}

void ProcessGroupCNCL::WorkCNCL::abort() {
  // Abort all communicators of this work
  cnclComm_->cnclCommAbort();
}

void ProcessGroupCNCL::WorkCNCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
}

bool ProcessGroupCNCL::WorkCNCL::isStarted() {
  checkAndSetException();
  return exception() || startedMLUExecutionInternal();
}

ProcessGroupCNCL::WorkCNCL::WorkCNCL(const WorkCNCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkCNCL>(w),
      device_(w.device_),
      cncl_start_event_(w.cncl_start_event_),
      cncl_end_event_(w.cncl_end_event_),
      cnclComm_(w.cnclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_),
      distDebugLevel_(w.distDebugLevel_) {
  exception_ = w.exception_;
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupCNCL continues supporting multi-process and multi-thread modes.";

ProcessGroupCNCL::ProcessGroupCNCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      traceKeyStart_(getTraceStartKey("CNCL", rank)),
      traceKeyEnd_(getTraceEndKey("CNCL", rank)),
      terminate_process_group_(false),
      terminateHeartbeatMonitorThread_(false),
      collectiveDebugInfoMode_(false),
      uid_(process_group_id++) {
  TORCH_CHECK(
      torch_mlu::device_count() != 0,
      "ProcessGroupCNCL is only supported with MLUs, no MLUs found!");

  logPrefix_ = createLogPrefix();
  blockingWait_ = getCvarBool(TORCH_CNCL_BLOCKING_WAIT, false);

  async_error_handling_ = static_cast<ErrorHandlingMode>(
      getCvarInt(TORCH_CNCL_ASYNC_ERROR_HANDLING, 0));

  desync_debug_ = getCvarBool(TORCH_CNCL_DESYNC_DEBUG, false) ||
      (dist_debug_level_ >= c10d::DebugLevel::Detail);
  monitorThreadEnabled_.store(getCvarBool(TORCH_CNCL_ENABLE_MONITORING, true));

  coordCheckIntervalMilSec_ = getCvarInt(TORCH_CNCL_COORD_CHECK_MILSEC, 1000);

  dumpOnTimeout_ = getCvarBool(TORCH_CNCL_DUMP_ON_TIMEOUT, false) ||
      (dist_debug_level_ >= c10d::DebugLevel::Detail);

  heartbeat_ = 1ULL;

  heartbeatTimeoutInSec_ =
      getCvarInt(TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC, 60 * 10 /*10 Mins*/);

  waitTimeoutDumpInMilSec_ =
      getCvarInt(TORCH_CNCL_WAIT_TIMEOUT_DUMP_MILSEC, 60 * 1000 /*60 Sec*/);
  cnclTraceBufferSize_ = getCvarInt(TORCH_CNCL_TRACE_BUFFER_SIZE, 0);
  CNCLTraceBuffer::get()->record_pg_ranks(pg_name_, groupRanks());

  // store_ usually is wrapped with c10d::PrefixStore and the prefix is
  // different across different ProcessGroupCNCL(PG) instances. We need to get
  // the underlying non-PrefixStore for sharing global information shared across
  // different PGs.
  c10d::PrefixStore* prefixStore =
      dynamic_cast<c10d::PrefixStore*>(store_.get());
  globalStore_ =
      prefixStore ? prefixStore->getUnderlyingNonPrefixStore() : store_;

  enableTiming_.store(
      getCvarBool(TORCH_CNCL_ENABLE_TIMING, false) || desync_debug_);

  avoidRecordStreams_ = getCvarBool(TORCH_CNCL_AVOID_RECORD_STREAMS, false);

  if (blockingWait_) {
    if (async_error_handling_ != NoHandling || desync_debug_) {
      LOG(INFO) << "[Rank " << rank_ << "] CNCL_BLOCKING_WAIT and "
                << "CNCL_ASYNC_ERROR_HANDLING|CNCL_DESYNC_DEBUG"
                << "should not both be enabled. "
                << "Only CNCL_BLOCKING_WAIT is being used in this process.";
      async_error_handling_ = NoHandling;
      desync_debug_ = false;
    }
  } else {
    if (desync_debug_ && async_error_handling_ == NoHandling) {
      LOG(INFO) << "[Rank " << rank_
                << "] CNCL_DESYNC_DEBUG and CNCL_ASYNC_ERROR_HANDLING "
                << "must both be enabled. "
                << "Enabling CNCL_ASYNC_ERROR_HANDLING.";
      async_error_handling_ = TearDown;
    }
  }

  cncl_comm_watchdog_thread_ =
      std::thread(&ProcessGroupCNCL::cnclCommWatchdog, this);
}

ProcessGroupCNCL::~ProcessGroupCNCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupCNCL destructor entered.";
  terminate_process_group_.store(true);

  workMetaListCV_.notify_one();

  if (cncl_comm_watchdog_thread_.joinable()) {
    cncl_comm_watchdog_thread_.join();
  }
  LOG(INFO) << logPrefix() << "ProcessGroupCNCL watchdog thread joined.";

  // Abort communicators after all threads have exited to avoid having the
  // threads dying due to aborted communicator and raising a SIGABRT
  // We need to include PG information in the abort reason so we can tell the
  // abort order.
  std::string abortReason = c10::str("Process Group destroyed on rank ", rank_);
  LOG(INFO)
      << logPrefix()
      << "ProcessGroupCNCL aborting communicators, check for 'abort finished' logs or look for abort hang";
  abort(abortReason);
  LOG(INFO) << logPrefix() << "ProcessGroupCNCL abort finished.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
  if (cnclHeartbeatMonitorThread_.joinable()) {
    cnclHeartbeatMonitorThread_.join();
  }

  {
    // Abort all CNCL Communicators on Process Group Destruction
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : dev_cncl_comm_map_) {
      auto& cncl_comm = it.second;
      cncl_comm->cnclCommAbort();
    }
  }
// gcov can not save the coverage data of the code run by subprocess,
// so we flush the coverge data manually
#ifdef TEST_COVERAGE
  __gcov_flush();
#endif
}

std::optional<std::function<std::string()>>& get_cpp_trace_dumper() {
  static std::optional<std::function<std::string()>> dumper(c10::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;
  std::future<bool> resultFuture = resultPromise.get_future();
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    try {
      auto& gil_checker = get_gil_checker();
      promise.set_value((*gil_checker)());
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  // Detach the thread to allow it to run independently
  workerThread.detach();

  return resultFuture;
}

int computeDeltaMS(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void ProcessGroupCNCL::heartbeatMonitor() {
  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitMsg;
  bool checkTimeoutSignal = (dumpOnTimeout_ && uid_ == 0);
  int monitorPollInterval = checkTimeoutSignal ? coordCheckIntervalMilSec_
                                               : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  std::optional<DumpPipe> dumpPipe = c10::nullopt;
  if (uid_ == 0) {
    // DumpPipe is one per-trainer process, and its convenient to name them
    // after 'global' ranks in the system, So we assume processgroup (uid)==0 is
    // the global PG and has globally unique rank ids across trainers.
    dumpPipe.emplace(rank_);
  }
  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }
    auto currentTime = std::chrono::steady_clock::now();

    // We put extra functionality in the thread for the default PG (aka, uid_=0)
    // because the signal is same across different PGs. We only need to run
    // once per process to avoid duplicate things performed in too many separate
    // threads. For example, we check a global flag on the TCPc10d::Store
    // periodically to see if any PG on any rank observed a timeout and signaled
    // peers to dump debugging info, and we avoid hammering the TCPc10d::Store
    // from all PGs on the same rank.
    if (checkTimeoutSignal) {
      // There are two scenarios where monitor thread will dump on timeout:
      // 1. The local rank is the first to observe a timeout.shouldDump_ will be
      // set to true.
      // 2. other ranks detected the timeout and signal the local rank to dump
      // In addtion, monitor threads will dump if watchdog threads has no
      // heartbeat or dumpPipe is not empty.
      if (shouldDump_.load()) {
        errorMsg = c10::str(
            logPrefix(),
            "Received a timeout signal from this local rank and will ",
            "start to dump the debug info. ",
            "Last enstreamd CNCL work: ",
            lastEnstreamdSeq_,
            ", last completed CNCL work: ",
            lastCompletedSeq_,
            ".");
        exitMsg = c10::str(
            "ProcessGroupCNCL's watchdog detected a collective timeout from the local rank. ",
            "This is most likely caused by incorrect usages of collectives, e.g., wrong ",
            "sizes used across ranks, the order of collectives is not same for all ranks ",
            "or the scheduled collective, for some reason, didn't run. Additionally, ",
            "this can be caused by GIL deadlock or other reasons such as network errors or ",
            "bugs in the communications library (e.g. CNCL), etc. We tried our best to ",
            "dump the debug info into the storage to help you debug the issue.");
        break;
      }
      // We poll store to see if some ranks have flagged a timeout when
      // we haven't polled for `heartbeat_timeout` seconds and there haven't
      // any work added or removed for `watchdog_timeout` seconds.
      if (computeDeltaMS(lastWorkListUpdateTime_, currentTime) >=
              k_watchdog_thread_sleep_millis &&
          computeDeltaMS(lastTimePollStore, currentTime) >=
              coordCheckIntervalMilSec_) {
        lastTimePollStore = currentTime;
        if (globalStore_->check({std::string(TIMEOUT_DUMP)})) {
          errorMsg = c10::str(
              logPrefix(),
              "Received a global timeout from another rank and will ",
              "start to dump the debug info. ",
              "Last enstreamd CNCL work: ",
              lastEnstreamdSeq_,
              ", last completed CNCL work: ",
              lastCompletedSeq_,
              ".");
          exitMsg = c10::str(
              "ProcessGroupCNCL's watchdog detected a collective timeout on some other rank and notified current rank. ",
              "This is most likely caused by incorrect usages of collectives, e.g., wrong ",
              "sizes used across ranks, the order of collectives is not same for all ranks ",
              "or the scheduled collective, for some reason, didn't run. Additionally, ",
              "this can be caused by GIL deadlock or other reasons such as network errors or ",
              "bugs in the communications library (e.g. CNCL), etc. We tried our best to ",
              "dump the debug info into the storage to help you debug the issue.");
          break;
        }
      }
    }

    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = heartbeat_.load();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        // No heartbeat increase detected and timeout.
        errorMsg = c10::str(
            logPrefix(),
            "Heartbeat monitor timed out! Process will be terminated after dumping debug info.",
            " workMetaList_.size()=",
            workMetaList_.size());
        exitMsg = c10::str(
            "ProcessGroupCNCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enstreamd collectives. ",
            "This typically indicates a CNCL/MLU API hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "MLU api, or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer timeout would help, ",
            "you can either increase the timeout (TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_CNCL_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the hang. "
            "workMetaList_.size() = ",
            workMetaList_.size(),
            "");
        break;
      }
    }
    // process a request to dump the trace. only PG uid 0 will respond to dump
    // requests, but this is fine since all PG's feed into the same flight
    // recorder and dump. After dump, the training should continue.
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // best effort dump, not waiting for the dump here
      std::future<bool> fut = std::async(
          std::launch::async, [this]() { return this->dumpDebuggingInfo(); });
    }
  }
  LOG(ERROR) << errorMsg;

  auto& cpp_dumper = get_cpp_trace_dumper();
  if (cpp_dumper.has_value()) {
    LOG(INFO) << "Dumping c++ stacktraces: " << cpp_dumper.value()();
  }

  // c10d::Store debug info to storage if no other thread does it. (By default
  // to local disk)
  std::future<bool> asyncDebugDump = std::async(
      std::launch::async, [this]() { return this->dumpDebuggingInfo(); });

  // wait for the dump until timeout
  waitForFutureOrTimeout(
      asyncDebugDump,
      std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
      "Flight recorder dump in heartbeatMonitor");

  if (get_gil_checker() != nullptr) {
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      TORCH_CHECK(
          futStatus != std::future_status::deferred,
          "Expected the future to have been launched eagerly.");
      LOG(ERROR)
          << "Could not acquire GIL within 300 ms on exit, possible GIL induced hang";
    }
    LOG(INFO) << "Could acquire GIL on exit";
  } else {
    LOG(INFO)
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> destructors
  // -> set terminateHeartbeatMonitorThread_ -> notify monitorWakeUpCV_.
  // So the code either early returns above or will skip the sleep below.
  // Case two: desync might be slow or get stuck. Or we get stuck in
  // destructors, we will sleep for some time before calling std::abort() to
  // kill the whole process.
  if ((terminate_process_group_.load() || collectiveDebugInfoMode_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    // Leave another two mins for desync report generation or process group
    // destroy.
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
  }

  // At this point, we either already sleep for another `heartbeatTimeoutInSec_`
  // or the thread has finished. Because we don't want to block the monitor
  // thread, so We mark the thread detach and the dump of debug info becomes
  // "best effort". If the process exit normally, marking it detach also makes
  // sense because we don't really care about dumping the debug info.

  // We already log completion inside the thread, so it may not be necessary to
  // check the return value here.  We mainly use a future so we can exit early
  // if done.

  if (!terminateHeartbeatMonitorThread_.load()) {
    // Create a error message reported from MonitorThread, so
    // we throw exception and make the whole process to be killed.
    // TODO: After having a hang debug wiki, we need to update the wiki
    // url here.
    const auto finalExitMsg = c10::str(logPrefix(), exitMsg);
    if (monitorThreadEnabled_.load()) {
      terminateProcess(finalExitMsg);
    } else {
      LOG(ERROR)
          << "PGCNCL Monitor Thread is disabled, but would have killed this job:\n"
          << finalExitMsg;
    }
  }
}

void ProcessGroupCNCL::terminateProcess(std::string errMsg) {
  // Logging with `FATAL`, after errMsg printed, it calls `std::abort()`
  // to terminate the program execution.
  LOG(FATAL) << logPrefix() << errMsg;
}

const int& ProcessGroupCNCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

const std::vector<uint64_t>& ProcessGroupCNCL::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && uid_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

// TODO: complete it when cncl support cnclCommDump
std::string dump_cncl_trace() {
  return CNCLTraceBuffer::get()->dump();
}

bool ProcessGroupCNCL::dumpDebuggingInfo() {
  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  LOG(ERROR) << logPrefix() << "ProcessGroupCNCL preparing to dump debug info.";
  if (cnclTraceBufferSize_ > 0) {
    // We dump cncl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto cnclTrace = dump_cncl_trace();
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    writer.write(cnclTrace);
    return true;
  }
  return false;
}

void ProcessGroupCNCL::cnclCommWatchdog() {
  try {
    VLOG(2) << logPrefix() << "Process group watchdog thread started!";
    cnclHeartbeatMonitorThread_ =
        std::thread(&ProcessGroupCNCL::heartbeatMonitor, this);
    watchdogHandler();
    VLOG(2) << logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      LOG(INFO)
          << logPrefix()
          << "main process destroyed cuda before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();

    } else {
      // Append error message reported from watchdogHandler
      const auto exitMsg = c10::str(
          logPrefix(),
          "Process group watchdog thread terminated with exception: ",
          e.what());
      LOG(ERROR) << exitMsg;
      // TODO clean up the rethrow - why is it stored in a class var and
      // rethrown?
      watchDogException_ =
          std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
      std::rethrow_exception(watchDogException_);
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

void ProcessGroupCNCL::watchdogHandler() {
  bool done = false;
  lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  auto lastStatusUpdateTime = std::chrono::steady_clock::now();
  std::list<ProcessGroupCNCL::WorkCNCL> done_workers;

  while (!done || !terminate_process_group_.load()) {
    {
      std::unique_lock<std::mutex> lock(workMetaList_mutex_);
      // We busy-poll the work vector every k_watchdog_thread_sleep_millis
      // milliseconds as long as the atomic is True.
      workMetaListCV_.wait_for(
          lock,
          std::chrono::milliseconds(k_watchdog_thread_sleep_millis),
          [&]() -> bool { return terminate_process_group_.load(); });
      // Bump up heart beat by one.
      heartbeat_++;

// Some versions of GLOG support less-spammy version of LOG_EVERY_MS
// in which case we don't want to spam the logs.
#ifdef LOG_EVERY_MS
      // Log the progress of this PG periodically
      C10_LOG_EVERY_MS(INFO, kWorkStatusUpdatePeriodMs) << c10::str(
          logPrefix(),
          "CNCL Work update periodically: ",
          "last enstreamd CNCL work: ",
          lastEnstreamdSeq_,
          ", last completed CNCL work: ",
          lastCompletedSeq_,
          ".");
#endif
      auto logger = ::c10d::C10dLogger::getLogger();
      if (logger &&
          computeDeltaMS(
              lastStatusUpdateTime, std::chrono::steady_clock::now()) >=
              kWorkStatusUpdatePeriodMs) {
        ::c10d::C10dLoggingData data;
        data.integers["pg_id"] = uid_;
        data.integers["rank"] = rank_;
        data.integers["global_rank"] = globalRank();
        data.integers["last_enstreamd_work"] = lastEnstreamdSeq_;
        data.integers["last_completed_work"] = lastCompletedSeq_;
        logger->log(data);
        lastStatusUpdateTime = std::chrono::steady_clock::now();
      }

      for (auto it = workMetaList_.begin(); it != workMetaList_.end();
           /* no increment */) {
        auto& work = *it;
        // When terminate_process_group_ is true, communicators have already
        // been aborted, So cannot check exception based on them. But watchdog
        // needs to finish the check for the works that have already been
        // enstreamd to workMetaList_
        if (!terminate_process_group_.load()) {
          work.checkAndSetException();
        }
        bool timedOut = work.checkTimeout();

        // If work hits an exception (either an error or timeout)
        if (work.exception()) {
          if (SHOULD_CLEAN_UP(async_error_handling_)) {
            // Abort work and corresponding communicators
            work.abort();
            // PG level abort, which would abort all other communicators on this
            // rank
            abort();
          }

          // Report desync state in case of timeout
          if (timedOut) {
            LOG(ERROR) << c10::str(
                logPrefix(),
                "Timeout at CNCL work: ",
                work.seq_,
                ", last enstreamd CNCL work: ",
                lastEnstreamdSeq_,
                ", last completed CNCL work: ",
                lastCompletedSeq_,
                ".");
            try {
              if (desync_debug_ || dumpOnTimeout_) {
                // Set shutdown mode, so the heartbeat monitor thread will not
                // abort process immediately.
                collectiveDebugInfoMode_.store(true);
                std::vector<uint8_t> vec(1);
                globalStore_->set(std::string(TIMEOUT_DUMP), vec);
              }

              if (dumpOnTimeout_) {
                // signal the monitor thread to start dumping
                shouldDump_.store(true);
                // This sleep is used to give time for dumping before throwing
                // exeption
                std::this_thread::sleep_for(
                    std::chrono::seconds(heartbeatTimeoutInSec_));
              }

              if (desync_debug_) {
                auto desyncMsg = getCNCLWatchdogDebugInfo();
                LOG(ERROR) << logPrefix() << desyncMsg;
              }
            } catch (const std::exception& e) {
              LOG(ERROR)
                  << logPrefix()
                  << "Failed to retrieve TORCH_CNCL_DESYNC_DEBUG report. "
                  << " Please file an issue. Error: " << e.what();
            } catch (...) {
              LOG(ERROR)
                  << logPrefix()
                  << "Failed to rerieve TORCH_CNCL_DESYNC_DEBUG report with unknown error."
                  << " Please file an issue.";
            }
          }
          // Throw exception
          work.handleException(async_error_handling_);
        }

        // Work status logging for desync debug
        if (desync_debug_) {
          if (work.isStarted()) {
            logWorkStart(work);
          }
          if (work.isCompleted()) {
            logWorkEnd(work);
          }
        }

        // Clean up completed work
        if (work.isCompleted()) {
          lastCompletedSeq_ = work.seq_;
          CNCLTraceBuffer::get()->retire_id(work.trace_id_, true);
          // TODO: support onCompletionHook_
          //  if (onCompletionHook_) {
          //    // Move Work object to completedWorkList_ to be consumed by the
          //    hook
          //    // thread
          //    {
          //      const std::lock_guard<std::mutex>
          //      lock(completedWorkListMutex_); completedWorkList_.splice(
          //          completedWorkList_.end(), workMetaList_, it++);
          //    }
          //    completedWorkListCV_.notify_one();
          //  } else {
          done_workers.push_back(std::move(*it));
          it = workMetaList_.erase(it);
          lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
          //  }

          torch_mlu::MLUGraph::dec_pending_event_queries();
        } else {
          // Increment the iterator if the current WorkCNCL object is not
          // completed.
          ++it;
        }
        // Increment heartbeat after each work processed,
        // in case processing is slowed down (but not hung) by cuda api
        // contention
        heartbeat_++;
      }
      done_workers.clear();
    }
    done = workMetaList_.empty();
  }
}

void ProcessGroupCNCL::logWorkStart(WorkCNCL& work) {
  if (work.startTraceUpdated_)
    return;

  if (terminate_process_group_.load() || storeError_)
    return;

  work.startTraceUpdated_ = true;
  storeError_ = !traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

void ProcessGroupCNCL::logWorkEnd(WorkCNCL& work) {
  if (terminate_process_group_.load() || storeError_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  storeError_ = !traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

std::string ProcessGroupCNCL::getCNCLWatchdogDebugInfo() {
  return retrieveDesyncReport(store_, "CNCL", rank_, size_);
}

std::string ProcessGroupCNCL::createLogPrefix() const {
  return c10::str("[PG ", uid_, " Rank ", rank_, "] ");
}

const std::string& ProcessGroupCNCL::logPrefix() const {
  return logPrefix_;
}

void ProcessGroupCNCL::abortCommsFromMap(
    std::unordered_map<std::string, std::shared_ptr<CNCLComm>>& cnclComms_map,
    std::optional<std::string> abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  for (auto& it : cnclComms_map) {
    auto& dev_name = it.first;
    auto& cncl_comm = it.second;

    LOG(INFO) << logPrefix() << "ProcessGroupCNCL destroying cncl_comm_ "
              << cncl_comm->cncl_comm_ << " on MLU device: " << dev_name;
    cncl_comm->cnclCommAbort(abortReason);

    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `cnclCommInit` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `cnclCommInit`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    c10::StreamId streamId = -1;
    if (cncl_streams_.find(dev_name) != cncl_streams_.end()) {
      auto stream = cncl_streams_.at(dev_name);
      streamId = stream.id();
    }

    LOG(INFO) << logPrefix() << "ProcessGroupCNCL destroyed "
              << " communicator on MLU device: " << dev_name
              << " with stream: " << streamId;
  }
}

// Abort all communicators on this rank
bool ProcessGroupCNCL::abort(std::optional<std::string> abortReason) {
  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(dev_cncl_comm_map_, abortReason);
  return true;
}

void ProcessGroupCNCL::shutdown() {
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminate_process_group_.store(true);
  workMetaListCV_.notify_one();

  std::string abortReason = c10::str("Process Group shutdown on rank ", rank_);
  // lauch abort asnd wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupCNCL abort asynchrounously.";
  std::future<bool> fut = std::async(std::launch::async, [this, abortReason]() {
    return this->abort(abortReason);
  });

  waitForFutureOrTimeout(fut, options_->timeout, "ProcessGroup abort");
  LOG(INFO) << logPrefix() << "ProcessGroupCNCL aborts successfully.";
}

void ProcessGroupCNCL::workEnstream(
    c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> work) {
  if (!terminate_process_group_.load()) {
    std::lock_guard<std::mutex> lock(workMetaList_mutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enstream work without outputs_.
    workMetaList_.emplace_back(WorkCNCL(*work));
  }
}

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> ProcessGroupCNCL::initWork(
    at::Device device,
    int rank,
    c10d::OpType opType,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs, // TODO: necessary?
    bool record) {
  auto r = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(
      device,
      rank,
      opType,
      seq_,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : c10::nullopt,
      desync_debug_,
      enableTiming_.load(),
      dist_debug_level_);
  if (record) {
    // Ideally record every work that we enstream, rather than every work we
    // create.
    // - at the time of this PR we do not currently enstream every created work
    // - but it is unsafe to steal refs to start/end cuda events from Works that
    //   may go out of scope before flight recorder has retired them,
    //   so we must ensure that any work that is initialized via initWork will
    //   be enstreamd
    // - initially, moved record() into workEnstream(), but found that makes it
    //   hard to get access to profilingTitle,
    //   inputs, and outputs for metadata recording, and we don't want to attach
    //   these objects to the Work becuase it has implications for keeping those
    //   tensors alive longer and adds overhead when copying Work objects
    //   between threads
    r->trace_id_ = CNCLTraceBuffer::get()->record(
        uid_,
        seq_,
        profilingTitle ? profilingTitle : "",
        inputs,
        outputs,
        r->cncl_start_event_.get(),
        r->cncl_end_event_.get());
  }
  return r;
}

void ProcessGroupCNCL::setSequenceNumberForGroup() {
} // CNCL just starts sequence numbers at 0.
  //
uint64_t ProcessGroupCNCL::getSequenceNumberForGroup() {
  return seq_;
}

void ProcessGroupCNCL::broadcastCNCLCliqueID(
    cnclCliqueId* cncl_id,
    const bool is_p2p_op = false,
    const std::string& p2p_key = "",
    const int p2p_rank = 0) {
  // For collective operations:
  // For every CNCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple CNCL communicators, so we use a sequence
  // number to differentiate between them.
  // For point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string store_key;
  if (!is_p2p_op) {
    store_key = std::to_string(cncl_comm_counter_++);
  } else {
    store_key = p2p_key;
  }
  if (rank_ == 0 || (is_p2p_op && p2p_rank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(cncl_id),
        reinterpret_cast<uint8_t*>(cncl_id) + CNCL_CLIQUE_ID_BYTES_SIZE);
    store_->set(store_key, vec);
  } else {
    try {
      auto vec = store_->get(store_key);
      if (vec.size() != CNCL_CLIQUE_ID_BYTES_SIZE) {
        throw std::runtime_error(
            "Unexpected CNCL clique ID length received "
            "from the store");
      }
      std::memcpy(cncl_id, vec.data(), vec.size());
    } catch (const std::exception& e) {
      std::string exception_msg = c10::str(
          "[",
          rank_,
          "] is setting up CNCL communicator and "
          "retreiving cnclCliqueId from [0] via c10d key-value store by key '",
          store_key,
          "', but store->get('",
          store_key,
          "') got error: ");
      TORCH_CHECK(false, exception_msg + e.what());
    } catch (...) {
      TORCH_CHECK(
          false,
          c10::str(
              "Unknown exception while [",
              rank_,
              "] is setting up CNCL communicator and "
              "retreiving cnclCliqueId from [0] via c10d "
              "key-value store by key '",
              store_key,
              "'"));
    }
  }
}

std::shared_ptr<CNCLComm> ProcessGroupCNCL::getCNCLComm(
    const std::string& device_key,
    at::Device& device,
    c10d::OpType op_type,
    int p2p_rank,
    bool is_send_recv_self) {
  // Sanity check
  if (device_key.empty()) {
    throw std::runtime_error(
        "Not able to create/get the CNCL Communicator since "
        "the MLU devices are not known");
  }

  if (bound_device_id_) {
    if (*bound_device_id_ != device) {
      LOG(ERROR) << logPrefix() << "Tensor found on device " << device
                 << " but backend constrained to " << *bound_device_id_;
      C10_THROW_ERROR(
          DistBackendError,
          "Attempt to perform collective on tensor not on device passed to init_process_group");
    }
  }

  usedDeviceIdxs_.insert(device.index());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (dev_cncl_comm_map_.find(device_key) != dev_cncl_comm_map_.end()) {
      // Reuse the cached communicator if there is one.
      return dev_cncl_comm_map_[device_key];
    }
  }

  // CNCL communicator not cached, create a new entry
  std::shared_ptr<CNCLComm> cncl_comm;

  // For batch_isend_irecv, cnclGroupStart() would be called upfront
  bool batch_p2p = cnclActiveGroupCounter_ > 0;
  bool single_p2p_op = c10d::isP2POp(op_type, batch_p2p);

  // Create the unique CNCL ID and broadcast it
  cnclCliqueId clique_id;

  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (single_p2p_op && p2p_rank == 0)) {
    C10D_CNCL_CHECK(cnclGetCliqueId(&clique_id), c10::nullopt);
  }

  // For point-to-point communication on the same process, don't need broadcast.
  if (!is_send_recv_self) {
    // Broadcast so that each process can have a unique CNCL ID
    broadcastCNCLCliqueID(&clique_id, single_p2p_op, device_key, p2p_rank);
  }

  torch_mlu::mlu::OptionalMLUGuard mlu_guard;

  // [Group Start/End Note] This is used to ensure that cncl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying CNCL calls will look like
  //   cnclGroupStart() // This is in batch_isend_irecv
  //   cnclGroupStart() // This is [Note 1]
  //   cnclInitComms() // Inside CNCLComm::create
  //   cnclSend()
  //   cnclGroupEnd() // This is [Note 2]
  //   cnclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the cncl communicator will be created in the last
  // cnclGroupEnd which means when cnclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active cncl groups to ensure cncl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (size_t i = 0; i < cnclActiveGroupCounter_; ++i) {
    C10D_CNCL_CHECK(cnclGroupEnd(), c10::nullopt);
  }

  // [Note 1] Create the CNCL communicators for each MLU
  C10D_CNCL_CHECK(cnclGroupStart(), c10::nullopt);

  // world size and rank
  int num_ranks, rank_id;
  if (!single_p2p_op) {
    // Collective, all-to-all, or batch P2P
    // One rank for each device
    num_ranks = getSize();
    rank_id = getRank();
  } else if (is_send_recv_self) {
    // Same process send and recv.
    num_ranks = 1;
    rank_id = 0;
  } else {
    // For point-to-point operation, there are only 2 processes involved so
    // the MLU rank is either 0 or 1.
    num_ranks = 2;
    rank_id = p2p_rank;
  }

  mlu_guard.set_index(device.index());

  if (!cncl_comm) {
    // Create the CNCL communicators for each MLU
    cncl_comm =
        CNCLComm::create(num_ranks, rank_id, device.index(), &clique_id);
  }

  // Create streams
  auto stream_val = torch_mlu::getStreamFromPool(
      options_->is_high_priority_stream, device.index());

  // [Note 2 ]
  C10D_CNCL_CHECK(cnclGroupEnd(), c10::nullopt);

  // See [Group Start/End Note]
  for (size_t i = 0; i < cnclActiveGroupCounter_; ++i) {
    C10D_CNCL_CHECK(cnclGroupStart(), c10::nullopt);
  }

  cncl_streams_.emplace(device_key, std::move(stream_val));
  updateCnclStream(&clique_id, cncl_streams_.at(device_key));

  cncl_events_.emplace(
      device_key, torch_mlu::MLUEvent(CNRT_NOTIFIER_DISABLE_TIMING_ALL));

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(mutex_);

  // Record the communicators based on cnclCliqueId.
  cncl_id_to_comm_map_.emplace(buildCnclUniqueIdStr(clique_id), cncl_comm);

  // Move the CNCL resource to cache
  dev_cncl_comm_map_.emplace(device_key, std::move(cncl_comm));
  return dev_cncl_comm_map_[device_key];
}

std::exception_ptr ProcessGroupCNCL::WorkCNCL::checkForCNCLErrors() {
  return checkForCNCLErrorsInternal(cnclComm_);
}

int64_t ProcessGroupCNCL::getCnclComm(int rankid) {
  auto device = getDeviceForRank(rankid);
  auto indexFromRank = device.index();
  auto indexFromCurrentDevice = torch_mlu::current_device();
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    if (indexFromRank != indexFromCurrentDevice) {
      std::string warning = "The indexFromRank " +
          std::to_string(indexFromRank) +
          ", is not equal indexFromCurrentDevice " +
          std::to_string(indexFromCurrentDevice) +
          ", which might be normal if the number of devices on your collective communication server is inconsistent. " +
          "you need to check if the current device is correct when calling the getCnclComm. " +
          "If it's wrong, it might have introduced an error.";
      TORCH_WARN(warning);
    }
  });
  const auto key = getKeyFromDevice(device);
  auto cnclComms = getCNCLComm(key, device, c10d::OpType::UNKNOWN);
  auto ret_cnclComm = cnclComms->getCnclComm();
  int64_t cncl_comm =
      static_cast<int64_t>(reinterpret_cast<intptr_t>(ret_cnclComm));
  return cncl_comm;
}

std::exception_ptr ProcessGroupCNCL::checkForCNCLErrors(
    std::shared_ptr<CNCLComm>& cncl_comm) {
  return checkForCNCLErrorsInternal(cncl_comm);
}

std::exception_ptr ProcessGroupCNCL::checkForCNCLErrorsInternal(
    std::shared_ptr<CNCLComm>& cncl_comm) {
  // Prioritize comm_failure_reason over checkForCnclError() result if
  // comm_failure_reason is set.
  auto comm_failure_reason = cncl_comm->getCnclCommFailureReason();
  if (comm_failure_reason != c10::nullopt) {
    return std::make_exception_ptr(std::runtime_error(c10::str(
        "CNCL communicator encountered error set by ProcessGroupCNCL: ",
        *comm_failure_reason)));
  }
  cnclResult_t cncl_async_err = cncl_comm->checkForCnclError();
  //  The return value of `cnclGetCommAsyncError` is
  //  semantically consistent with the `cnclCommGetAsyncError` parameter.
  if (cncl_async_err != CNCL_RET_SUCCESS &&
      cncl_async_err != CNCL_RET_ASYNC_IN_PROGRESS) {
    return std::make_exception_ptr(std::runtime_error(
        "CNCL error: " + getCnclErrorDetailStr(cncl_async_err)));
  }

  return nullptr;
}
namespace {

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

// Check validity of tensor
void check_mlu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.device().is_privateuseone() || tensor.is_sparse()) {
    throw std::runtime_error("Tensors must be MLU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    throw std::runtime_error("Tensors must be contiguous");
  }
}

int64_t check_mlu_tensors_same_device(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.device().is_privateuseone() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be MLU and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK_WITH(
        ValueError,
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
}

at::Tensor newLikeFlat(std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    TORCH_CHECK(false, "Received an empty list");
  }
  auto& t = tensors[0];
  at::DeviceGuard mluGuard(t.device());
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}
} // namespace

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::collectiveCoalesced(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    c10d::OpType op_type,
    const char* profilingTitle,
    bool avoid_record_streams) {
  // Environment setting by the user may add onto collective call's option
  avoid_record_streams |= avoidRecordStreams_;
  torch_mlu::CaptureStatus capture_status =
      torch_mlu::currentStreamCaptureStatusMayInitCtx();

  seq_++;

  auto device = getDevice(inputs[0]);
  const auto key = getKeyFromDevice(device);

  auto cncl_comm = getCNCLComm(key, device, op_type);

  auto cncl_stream = cncl_streams_.at(key);

  // First let CNCL stream wait for input tensors allocation stream
  syncStream(device, cncl_events_[key], cncl_stream);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = cncl_comm;
    } else {
      TORCH_CHECK(coalescedComm_ == cncl_comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  auto work = initWork(
      device, rank_, op_type, profilingTitle, inputs, outputs, /*record=*/true);

  // c10d::Store references to outputs to be used by WorkCNCL::result and
  // operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoid_record_streams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  torch_mlu::mlu::OptionalMLUGuard mlu_guard;

  // Start event should only be recorded before the cnclGroupStart()
  if (desync_debug_) {
    work->cncl_start_event_->place(cncl_stream);
  }

  {
    AutoCnclGroup cncl_group_guard;
    for (const auto i : c10::irange(inputs.size())) {
      // Both `inputs' and `outputs' are created on a worker stream and used in
      // different cnclStreams.  Hence, both must record the cncl_stream to
      // prevent being freed before the collective finishes.
      //
      // We only record `inputs' here, and leave recording `outputs' to `fn' for
      // operations where `inputs' and `outputs' are not the same.
      if (!avoid_record_streams) {
        torch_mlu::MLUCachingAllocator::recordStream(
            inputs[i].storage().data_ptr(), cncl_stream);
      }
      C10D_CNCL_CHECK(
          fn(inputs[i], outputs[i], cncl_comm->getCnclComm(), cncl_stream),
          cncl_comm->getCnclCommFailureReason());
    }
  }

  work->cncl_end_event_->place(cncl_stream);
  work->cnclComm_ = cncl_comm;

  {
    // Set current stream to init future's event with cncl stream
    torch_mlu::mlu::MLUMultiStreamGuard stream_guard(cncl_stream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  work->blockingWait_ = blockingWait_;
  work->opTimeout_ = options_->timeout;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->store_ = store_;

  // Notify graphs before we check the capture status preemptively
  torch_mlu::MLUGraph::inc_pending_event_queries();

  if (!coalescing_state_ && capture_status == torch_mlu::CaptureStatus::None) {
    workEnstream(work);
  } else {
    torch_mlu::MLUGraph::dec_pending_event_queries();
  }

  return work;
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    c10d::OpType op_type,
    const char* profilingTitle,
    bool avoid_record_streams) {
  // Environment setting by the user may add onto collective call's option
  avoid_record_streams |= avoidRecordStreams_;
  torch_mlu::CaptureStatus capture_status =
      torch_mlu::currentStreamCaptureStatusMayInitCtx();
  seq_++;

  auto device = getDevice(input);
  const auto key = getKeyFromDevice(device);

  auto cncl_comm = getCNCLComm(key, device, op_type);

  auto cncl_stream = cncl_streams_.at(key);

  // First let CNCL stream wait for input tensors allocation stream
  syncStream(device, cncl_events_[key], cncl_stream);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = cncl_comm;
    } else {
      TORCH_CHECK(coalescedComm_ == cncl_comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  std::vector<at::Tensor> inputs{input};
  std::vector<at::Tensor> outputs{output};

  bool enstream =
      !coalescing_state_ && capture_status == torch_mlu::CaptureStatus::None;
  auto work = initWork(
      device, rank_, op_type, profilingTitle, inputs, outputs, enstream);

  // c10d::Store references to outputs to be used by WorkCNCL::result and
  // operator<<.
  work->outputs_ =
      std::make_shared<std::vector<at::Tensor>>(std::move(outputs));

  if (avoid_record_streams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
    work->stashed_for_allocator_safety_->push_back(input);
  }

  torch_mlu::mlu::OptionalMLUGuard mlu_guard;

  // Start event should only be recorded before the cnclGroupStart()
  if (desync_debug_) {
    work->cncl_start_event_->place(cncl_stream);
  }

  pre(cncl_stream, work);
  // Both `inputs' and `outputs' are created on a worker stream and used in
  // different cncl stream.  Hence, both must record the cncl stream to
  // prevent being freed before the collective finishes.
  //
  // We only record `inputs' here, and leave recording `outputs' to `fn' for
  // operations where `inputs' and `outputs' are not the same.
  //
  // See [Sync Streams].
  {
    AutoCnclGroup cncl_group_guard;
    if (!avoid_record_streams) {
      torch_mlu::MLUCachingAllocator::recordStream(
          input.storage().data_ptr(), cncl_stream);
    }
    C10D_CNCL_CHECK(
        fn(input, output, cncl_comm->getCnclComm(), cncl_stream),
        cncl_comm->getCnclCommFailureReason());
  }

  post(cncl_stream, work);

  // End event should only be recorded after the cnclGroupEnd()
  if (!coalescing_state_) {
    work->cncl_end_event_->place(cncl_stream);
  }
  work->cnclComm_ = cncl_comm;

  {
    // Set current stream to init future's event with cncl stream
    torch_mlu::mlu::MLUMultiStreamGuard stream_guard(cncl_stream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  work->blockingWait_ = blockingWait_;
  work->opTimeout_ = options_->timeout;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->store_ = store_;

  // Notify graphs before we check the capture status preemptively
  torch_mlu::MLUGraph::inc_pending_event_queries();
  if (enstream) {
    workEnstream(work);
  } else {
    torch_mlu::MLUGraph::dec_pending_event_queries();
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::collective(
    at::Tensor& input,
    at::Tensor& output,
    Fn fn,
    c10d::OpType op_type,
    const char* profilingTitle,
    bool avoid_record_streams) {
  return collective(
      input,
      output,
      fn,
      [](torch_mlu::MLUStream&,
         c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
      [](torch_mlu::MLUStream&,
         c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
      op_type,
      profilingTitle,
      avoid_record_streams);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_mlu_single_tensor(tensor);
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = mlu_data_ptr(input_impl);
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = mlu_data_ptr(output_impl);
        return cnclAllReduce(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            getCnclReduceOp(opts.reduceOp, input),
            comm,
            stream.stream());
      },
      c10d::OpType::ALLREDUCE,
      "cncl:all_reduce");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts) {
  auto total_numel = check_mlu_tensors_same_device(tensors);

  return collectiveCoalesced(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        auto cnclDataType = getCnclDataType(input.scalar_type());
        auto cnclReduceOp = getCnclReduceOp(opts.reduceOp, input);
        return cnclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            cnclDataType,
            cnclReduceOp,
            comm,
            stream.stream());
      },
      c10d::OpType::COALESCED,
      "cncl:allreduce_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_mlu_single_tensor(tensor);

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = mlu_data_ptr(input_impl);
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = mlu_data_ptr(output_impl);
        const int root = opts.rootRank + opts.rootTensor;
        return cnclBroadcast(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      c10d::OpType::BROADCAST,
      "cncl:broadcast");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::reduce(
    std::vector<at::Tensor>& tensors,
    const c10d::ReduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_mlu_single_tensor(tensor);

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = mlu_data_ptr(input_impl);
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = mlu_data_ptr(output_impl);
        const int root = opts.rootRank + opts.rootTensor;
        return cnclReduce(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            getCnclReduceOp(opts.reduceOp, input),
            root,
            comm,
            stream.stream());
      },
      c10d::OpType::REDUCE,
      "cncl:reduce");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::
    allgather_into_tensor_coalesced(
        std::vector<at::Tensor>& outputs,
        std::vector<at::Tensor>& inputs,
        const c10d::AllgatherOptions& opts) {
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        return cnclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getCnclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      c10d::OpType::COALESCED,
      "cncl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(input_tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);

  auto input_tensor = input_tensors.back();
  check_mlu_single_tensor(input_tensor);
  auto output_tensors_ = output_tensors.back();

  bool same_size = check_same_size(output_tensors_);
  if (same_size) {
    // check if origin output tensors are flattened
    bool flattened = checkTensorsNeedFlatten(output_tensors_);

    // Flatten a vector of tensors into a single, stacked tensor.
    Tensor output_flattened;
    if (!flattened) {
      output_flattened = newLikeFlat(output_tensors_);
    } else {
      output_flattened = output_tensors_[0];
    }
    return collective(
        input_tensor,
        output_flattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            cnclComm_t comm,
            torch_mlu::MLUStream& stream) {
          auto input_impl = torch_mlu::getMluTensorImpl(input);
          auto input_ptr = mlu_data_ptr(input_impl);
          auto output_impl = torch_mlu::getMluTensorImpl(output);
          auto output_ptr = mlu_data_ptr(output_impl);
          if (!avoidRecordStreams_) {
            torch_mlu::MLUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return cnclAllGather(
              input_ptr,
              output_ptr,
              input.numel(),
              getCnclDataType(input.scalar_type()),
              comm,
              stream.stream());
        },
        [&](torch_mlu::MLUStream& cncl_stream,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
        [&](torch_mlu::MLUStream& cncl_stream,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
          // Copy the flattened output tensors to the outputs.
          torch_mlu::mlu::MLUStreamGuard guard(cncl_stream);
          for (const auto j : c10::irange(output_tensors_.size())) {
            if (!avoidRecordStreams_) {
              torch_mlu::MLUCachingAllocator::recordStream(
                  output_tensors_[j].storage().data_ptr(), cncl_stream);
            }
            if (!flattened)
              output_tensors_[j].copy_(output_flattened[j], true);
          }
        },
        c10d::OpType::ALLGATHER,
        "cncl:all_gather");
  } else {
    const auto num_reduces = output_tensors_.size();
    startCoalescing();
    for (const int i : c10::irange(num_reduces)) {
      auto& output = output_tensors_[i];
      auto& input = (i == rank_) ? input_tensor : output;
      auto broadcastOpts = c10d::BroadcastOptions{
          static_cast<int64_t>(i), static_cast<int64_t>(0), opts.timeout};
      _broadcast_oop(output, input, broadcastOpts);
    }
    auto work = endCoalescing(c10d::OpType::ALLGATHER);
    return work;
  }
}

// _broadcast_oop adds an out-of-place broadcast in PGCNCL
// Custom collectives may be implemented by coalescing broadcast operations
// One use-case is implementing a vector all_gather (all_gather_v)
// where unevenly sized inputs are gathered among participating ranks
// Since all_gather provides an out-of-place API, an all_gather_v
// semantic implemented inside pg_cncl.all_gather also needs to support
// out-of-place, for which an out-of-place broadcast is required to be added
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_broadcast_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::BroadcastOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }

  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        return cnclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            getCnclDataType(input.scalar_type()),
            root,
            comm,
            stream.stream());
      },
      c10d::OpType::BROADCAST,
      "cncl:_broadcast_oop");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10d::AllgatherOptions& opts) {
  check_mlu_single_tensor(input_tensor);
  check_mlu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "output tensor size must be equal to world_size times input tensor size");
  }

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = mlu_data_ptr(input_impl);
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = mlu_data_ptr(output_impl);
        if (!avoidRecordStreams_) {
          torch_mlu::MLUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return cnclAllGather(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            comm,
            stream.stream());
      },
      c10d::OpType::_ALLGATHER_BASE,
      "cncl:all_gather_base");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

// _reduce_oop exposes an out-of-place reduce from PGCNCL
// Custom collectives may be implemented by coalescing reduce operations
// One use-case is implementing a vector reduce_scatter (reduce_scatter_v)
// where inputs are reduced and scattered unevenly among participating ranks
// Since reduce_scatter provides an out-of-place API, a reduce_scatter_v
// semantic implemented inside pg_cncl.reduce_scatter also needs to support
// out-of-place, for which an out-of-place reduce is required to be added
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_reduce_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const c10d::ReduceOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }

  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        const auto root = opts.rootRank + opts.rootTensor;
        const auto cncl_data_type = getCnclDataType(input.scalar_type());
        const auto cncl_reduce_op = getCnclReduceOp(opts.reduceOp, input);
        return cnclReduce(
            input.data_ptr(),
            output.data_ptr(),
            input.numel(),
            cncl_data_type,
            cncl_reduce_op,
            (int)root,
            comm,
            stream.stream());
      },
      c10d::OpType::REDUCE,
      "cncl:_reduce_oop");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(output_tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);

  auto output_tensor = output_tensors.back();
  check_mlu_single_tensor(output_tensor);
  auto input_tensors_ = input_tensors.back();

  bool same_size = check_same_size(input_tensors_);
  if (same_size) {
    // check if origin input tensors are flattened
    bool flattened = checkTensorsNeedFlatten(input_tensors_);

    // Flatten a vector of tensors into a single, stacked tensor.
    Tensor input_flattened;
    if (!flattened) {
      input_flattened = newLikeFlat(input_tensors_);
    } else {
      input_flattened = input_tensors_[0];
    }

    return collective(
        input_flattened,
        output_tensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            cnclComm_t comm,
            torch_mlu::MLUStream& stream) {
          auto input_impl = torch_mlu::getMluTensorImpl(input);
          auto input_ptr = mlu_data_ptr(input_impl);
          auto output_impl = torch_mlu::getMluTensorImpl(output);
          auto output_ptr = mlu_data_ptr(output_impl);
          if (!avoidRecordStreams_) {
            torch_mlu::MLUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return cnclReduceScatter(
              input_ptr,
              output_ptr,
              output.numel(),
              getCnclDataType(input.scalar_type()),
              getCnclReduceOp(opts.reduceOp, input),
              comm,
              stream.stream());
        },
        [&](torch_mlu::MLUStream& cncl_stream,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
          if (avoidRecordStreams_) {
            // We only need to stash input_tensors.
            //  - inputFlattened is stashed onto
            //  work->stashed_for_allocator_safety_
            //    in collective().
            //  - User-facing outputTensors is stashed onto work->outputs_
            //  in collective(),
            //    and should also be held by the user until after waiting on
            //    work_.
            auto& v = work->stashed_for_allocator_safety_;
            v->insert(v->end(), input_tensors_.begin(), input_tensors_.end());
          }
          // Copy the input tensors to the flattened inputs.
          torch_mlu::mlu::MLUStreamGuard guard(cncl_stream);
          for (const auto j : c10::irange(input_tensors_.size())) {
            if (!avoidRecordStreams_) {
              torch_mlu::MLUCachingAllocator::recordStream(
                  input_tensors_[j].storage().data_ptr(), cncl_stream);
            }
            if (!flattened)
              input_flattened[j].copy_(input_tensors_[j], true);
          }
        },
        [&](torch_mlu::MLUStream&,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
        c10d::OpType::REDUCE_SCATTER,
        "cncl:reduce_scatter");
  } else {
    const auto num_reduces = input_tensors_.size();
    startCoalescing();
    for (const int i : c10::irange(num_reduces)) {
      auto& input = input_tensors_[i];
      auto& output = (i == rank_) ? output_tensor : input;
      auto reduceOpts = c10d::ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i),
          static_cast<int64_t>(0),
          opts.timeout};
      _reduce_oop(output, input, reduceOpts);
    }
    auto work = endCoalescing(c10d::OpType::REDUCE_SCATTER);
    return work;
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::
    reduce_scatter_tensor_coalesced(
        std::vector<at::Tensor>& outputs,
        std::vector<at::Tensor>& inputs,
        const c10d::ReduceScatterOptions& opts) {
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        if (!avoidRecordStreams_) {
          torch_mlu::MLUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        auto cnclDataType = getCnclDataType(input.scalar_type());
        auto cnclReduceOp = getCnclReduceOp(opts.reduceOp, input);
        return cnclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            output.numel(),
            cnclDataType,
            cnclReduceOp,
            comm,
            stream.stream());
      },
      c10d::OpType::COALESCED,
      "cncl:reduce_scatter_tensor_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_reduce_scatter_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10d::ReduceScatterOptions& opts) {
  if (input_tensor.dtype() != output_tensor.dtype()) {
    C10_THROW_ERROR(
        TypeError, "input tensor must be the same type as the output tensor.");
  }

  if (input_tensor.numel() != output_tensor.numel() * size_) {
    C10_THROW_ERROR(
        ValueError,
        "input tensor must be the same size as output size times world size");
  }

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = mlu_data_ptr(input_impl);
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = mlu_data_ptr(output_impl);
        if (!avoidRecordStreams_) {
          torch_mlu::MLUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return cnclReduceScatter(
            input_ptr,
            output_ptr,
            output.numel(),
            getCnclDataType(input.scalar_type()),
            getCnclReduceOp(opts.reduceOp, input),
            comm,
            stream.stream());
      },
      c10d::OpType::_REDUCE_SCATTER_BASE,
      "cncl:reduce_scatter");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::barrier(
    const c10d::BarrierOptions& opts) {
  std::vector<at::Device> devices;
  if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a CNCL collective being called
    // Here we have to use the best guesses and will use a single MLU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different MLU
    auto num_mlus = torch_mlu::device_count();
    int16_t device_idx = static_cast<int16_t>(rank_ % num_mlus);
    LOG(INFO) << c10::str(
        "Rank ",
        this->getRank(),
        " using MLU ",
        device_idx,
        " to perform barrier as devices used by this process are currently unknown. ",
        "This can potentially cause a hang if this rank to MLU mapping is incorrect.",
        "Specify device_ids in barrier() to force use of a particular device.");
    devices.push_back(at::Device(at::DeviceType::PrivateUse1, device_idx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.push_back(at::Device(at::DeviceType::PrivateUse1, usedDeviceIdx));
    }
  }

  // Use one device only
  auto device = devices.back();
  std::vector<at::Tensor> barrierTensors;
  barrierTensors.push_back(
      at::empty({1}, at::TensorOptions().device(device).dtype(at::kFloat)));

  // All reduce to achieve the barrier
  auto work = allreduce(barrierTensors);

  // Work will take over barrierTensors
  auto cncl_work = dynamic_cast<ProcessGroupCNCL::WorkCNCL*>(work.get());
  TORCH_CHECK(cncl_work);
  cncl_work->barrierTensor_ = std::move(barrierTensors[0]);

  return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::alltoall_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    std::vector<int64_t>& output_split_sizes,
    std::vector<int64_t>& input_split_sizes,
    const c10d::AllToAllOptions& /* unused */) {
  check_mlu_single_tensor(output_tensor);
  check_mlu_single_tensor(input_tensor);

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = mlu_data_ptr(input_impl);
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = mlu_data_ptr(output_impl);
        auto dtype = getCnclDataType(input.scalar_type());

        // When equal split, use cnclAlltoAll to improve performance
        if (output_split_sizes.size() == 0 && input_split_sizes.size() == 0) {
          int64_t cnt = input.numel() / size_;
          if (cnt == 0) {
            return CNCL_RET_SUCCESS;
          }
          if (!avoidRecordStreams_) {
            torch_mlu::MLUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }

          return cnclAlltoAll(
              input_ptr, output_ptr, cnt, dtype, comm, stream.stream());
        } else {
          c10d::checkSplitSizes(input_split_sizes, input, size_);
          c10d::checkSplitSizes(output_split_sizes, output, size_);
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);

          c10d::computeLengthsAndOffsets(
              input_split_sizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              output_split_sizes, output, &recv_lengths, &recv_offsets);

          size_t input_elem_size = getCnnlTypeSize(getCnnlType(input_impl));

          if (!avoidRecordStreams_) {
            torch_mlu::MLUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }

          torch_mlu::cncl::detail::all2all_single_unequal_split(
              input_ptr,
              send_lengths.data(),
              send_offsets.data(),
              output_ptr,
              recv_lengths.data(),
              recv_offsets.data(),
              input_elem_size,
              input.scalar_type(),
              comm,
              stream);
        }
        return CNCL_RET_SUCCESS;
      },
      c10d::OpType::ALLTOALL_BASE,
      "cncl:AlltoAll");
}

ProcessGroupCNCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(CNCL_BACKEND_NAME),
      is_high_priority_stream(is_high_priority_stream) {}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllToAllOptions& /* unused */) {
  TORCH_CHECK(
      input_tensors.size() == size_ && output_tensors.size() == size_,
      "Size of input tensor list not equal group size");

  auto device = output_tensors[0].device();
  for (size_t r = 0; r < output_tensors.size(); r++) {
    check_mlu_single_tensor(output_tensors[r]);
    check_mlu_single_tensor(input_tensors[r]);
    TORCH_CHECK(
        device == output_tensors[r].device() &&
            device == input_tensors[r].device(),
        "Tensors must be on the same device");
  }

  return collective(
      input_tensors[0],
      output_tensors[0],
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        torch_mlu::cncl::detail::all2all(
            output_tensors, input_tensors, comm, stream);
        return CNCL_RET_SUCCESS;
      },
      [&](torch_mlu::MLUStream& cncl_stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        if (avoidRecordStreams_) {
          // input_tensor0 and output_tensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), input_tensors.begin(), input_tensors.end());
          v->insert(v->end(), output_tensors.begin(), output_tensors.end());
        }
      },
      [&](torch_mlu::MLUStream& cncl_stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
      c10d::OpType::ALLTOALL,
      "cncl:AlltoAllv");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupCNCL::gather: " + msg);
  };

  c10d::assertRootRank(invalidArgument, opts.rootRank, size_);
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);

  // @lint-ignore CLANGTIDY
  auto tensor = inputTensors.back();

  std::vector<at::Tensor> outputs;

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = tensor.options();
    const auto& sizes = tensor.sizes();
    c10d::assertTypeAndSizesMatch(
        invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (outputTensors.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  // avoidRecordStreams_ note: collective() will stash inputTensors and
  // outputs, which == outputTensors[0] on the root rank where it matters.
  return collective(
      tensor,
      outputs[0],
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoidRecordStreams_) {
            for (auto output : outputs) {
              torch_mlu::MLUCachingAllocator::recordStream(
                  output.storage().data_ptr(), stream);
            }
          }
        }
        torch_mlu::cncl::detail::gather(tensor, outputs, comm, stream, root);
        return CNCL_RET_SUCCESS;
      },
      c10d::OpType::GATHER,
      "cncl:gather");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupCNCL::scatter: " + msg);
  };

  c10d::assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto outputTensor = outputTensors.back();

  std::vector<at::Tensor> inputs;

  if (getRank() == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = outputTensor.options();
    const auto& sizes = outputTensor.sizes();
    c10d::assertTypeAndSizesMatch(
        invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
    if (inputTensors.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  // avoidRecordStreams_ note: collective() will stash outputTensors and
  // inputs, which == inputTensors[0] on the root rank where it matters.
  bool avoid_record_streams = avoidRecordStreams_ || (!opts.asyncOp);

  // TODO:[PYTORCH-11352]
  return collective(
      outputTensor,
      inputs[0], // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoid_record_streams) {
            for (auto input : inputs) {
              torch_mlu::MLUCachingAllocator::recordStream(
                  input.storage().data_ptr(), stream);
            }
          }
        }
        torch_mlu::cncl::detail::scatter(
            inputs, outputTensor, comm, stream, root);
        return CNCL_RET_SUCCESS;
      },
      c10d::OpType::SCATTER,
      "cncl:scatter",
      avoid_record_streams);
}

void ProcessGroupCNCL::groupStart() {
  C10D_CNCL_CHECK(cnclGroupStart(), c10::nullopt);
  ++cnclActiveGroupCounter_;
}

void ProcessGroupCNCL::groupEnd() {
  C10D_CNCL_CHECK(cnclGroupEnd(), c10::nullopt);
  --cnclActiveGroupCounter_;
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    c10d::OpType op_type,
    const char* profilingTitle) {
  // avoidRecordStreams_ note:
  // send, recv, and irecv should be ok with avoid_record_streams,
  // However, for isend, I don't think the API requires the user
  // to wait() on the returned handle, so ProcessGroupCNCL can't know
  // when it's safe to release the input back to the allocator,
  // and the present call has no way to know it's not an isend.
  // Therefore, we warn and fall back to the typical recordStream logic:
  TORCH_WARN_ONCE(
      avoidRecordStreams_,
      "CNCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point "
      "collectives.");

  std::string key;
  auto device = getDevice(tensor);

  int p2p_rank = 0, p2p_target_rank = 0;
  bool is_send_recv_self = false;

  // For batch_isend_irecv, cnclGroupStart() would be called upfront
  bool batchP2P = cnclActiveGroupCounter_ > 0;

  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevice(device);
    p2p_rank = rank_;
    p2p_target_rank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2p_rank = rank_ <= peer ? 0 : 1;
    is_send_recv_self = rank_ == peer;
    p2p_target_rank = is_send_recv_self ? 0 : 1 - p2p_rank;

    if (!coalescing_state_) {
      // Bump sequence number. Don't do so if it's a batch P2P, it will be
      // bumped in `endCoalescing`.
      seq_++;
    }
  }

  auto cncl_comm =
      getCNCLComm(key, device, op_type, p2p_rank, is_send_recv_self);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = cncl_comm;
    } else {
      TORCH_CHECK(coalescedComm_ == cncl_comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  auto cncl_stream = cncl_streams_.at(key);
  syncStream(device, cncl_events_[key], cncl_stream);

  auto work = initWork(
      device, rank_, op_type, profilingTitle, {tensor}, {}, /*record=*/false);
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
  work->outputs_->push_back(tensor);

  torch_mlu::mlu::OptionalMLUGuard mlu_guard;

  // Start event should only be recorded before the cnclGroupStart()
  if (desync_debug_) {
    work->cncl_start_event_->place(cncl_stream);
  }

  torch_mlu::MLUCachingAllocator::recordStream(
      tensor.storage().data_ptr(), cncl_stream);

  {
    AutoCnclGroup cncl_group_guard;
    cnclComm_t comm_ = cncl_comm->getCnclComm();
    C10D_CNCL_CHECK(
        fn(tensor, comm_, cncl_stream, p2p_target_rank),
        cncl_comm->getCnclCommFailureReason());
  }

  // End event should only be recorded after the cnclGroupEnd()
  if (!coalescing_state_) {
    work->cncl_end_event_->place(cncl_stream);
  }
  work->cnclComm_ = cncl_comm;

  work->blockingWait_ = blockingWait_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;

  {
    // Set current stream to init future's event with cncl stream
    torch_mlu::mlu::MLUMultiStreamGuard stream_guard(cncl_stream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    // Set current stream to init future's event with cncl stream
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  if (work->recordFunctionEndCallback_) {
    work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
      work->recordFunctionEndCallback_();
    });
  }

  // Enstream P2P op so that it can be cancelled by CNCL watchdog
  torch_mlu::CaptureStatus capture_status =
      torch_mlu::currentStreamCaptureStatusMayInitCtx();

  // Notify graphs before we check the capture status preemptively
  torch_mlu::MLUGraph::inc_pending_event_queries();

  if (!coalescing_state_ && capture_status == torch_mlu::CaptureStatus::None) {
    workEnstream(work);
    return work;
  } else {
    torch_mlu::MLUGraph::dec_pending_event_queries();
    return nullptr;
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::send(
    std::vector<at::Tensor>& tensors,
    int dst_rank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_mlu_single_tensor(tensor);
  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          int dst) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        return cnclSend(
            mlu_data_ptr(input_impl),
            input.numel(),
            getCnclDataType(input.scalar_type()),
            dst,
            comm,
            stream.stream());
      },
      dst_rank,
      c10d::OpType::SEND,
      "cncl:send");
  return ret;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::recv(
    std::vector<at::Tensor>& tensors,
    int src_rank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  check_mlu_single_tensor(tensor);
  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          int src) {
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        return cnclRecv(
            mlu_data_ptr(output_impl),
            output.numel(),
            getCnclDataType(output.scalar_type()),
            src,
            comm,
            stream.stream());
      },
      src_rank,
      c10d::OpType::RECV,
      "cncl:recv");
  return ret;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupCNCL::WorkCNCL::
    getFuture() {
  return future_;
}

void ProcessGroupCNCL::startCoalescing() {
  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescing_state_ |= CoalActive;
  groupStart();
  seq_++;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::endCoalescing() {
  // Default c10d::OpType to COALESCED if not specified
  return endCoalescing(c10d::OpType::COALESCED);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::endCoalescing(
    c10d::OpType op_type) {
  if (coalescedComm_ == nullptr) {
    // There is no actual work being coalesced, return here
    groupEnd();
    coalescing_state_ = 0;
    return nullptr;
  }

  TORCH_CHECK(
      coalescedDevice_.index() >= 0,
      "Somthing went wrong. Did you call end_coalescing before start_coalescing?");

  // `coalescedComm_` should have same set of comms across collectives
  auto comm = coalescedComm_;
  // `coalescedDevices_` should have same set of devices across collectives
  auto device = coalescedDevice_;

  // `getKeyFromDevice` is how we get keys for both collectives and batch P2P
  const auto key = getKeyFromDevice(device);
  auto& cncl_stream = cncl_streams_.at(key);

  // Bump collective counter
  seq_++;

  // Create Work object
  torch_mlu::CaptureStatus capture_status =
      torch_mlu::currentStreamCaptureStatusMayInitCtx();
  bool enstream =
      (coalescing_state_) && capture_status == torch_mlu::CaptureStatus::None;
  auto work =
      initWork(device, rank_, op_type, "cncl:coalesced", {}, {}, enstream);

  work->cnclComm_ = comm;
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams_;
  work->opTimeout_ = options_->timeout;
  work->store_ = store_;

  groupEnd();

  work->cncl_end_event_->place(cncl_stream);

  if (avoidRecordStreams_) {
    // other functions expect an initialized ptr if avoidRecordStreams_ is set
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }

  // Notify graphs before we check the capture status preemptively
  torch_mlu::MLUGraph::inc_pending_event_queries();

  if (enstream) {
    workEnstream(work);
  } else {
    torch_mlu::MLUGraph::dec_pending_event_queries();
  }

  coalescing_state_ = 0;

  return work;
}

void ProcessGroupCNCL::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    bool throwException) {
  std::string errorMsg;
  TORCH_CHECK(fut.valid(), "Expected a valid future");
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // Calling .get() will re-raise any exception from the future, and we don't
    // care about the retval
    try {
      bool result = fut.get();
      if (result) {
        LOG(INFO) << logPrefix()
                  << "future is successfully executed for: " << futDescription;
      }
    } catch (const std::exception& e) {
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waitng for future ",
          futDescription,
          ": ",
          e.what());
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waitng for future ",
          futDescription);
      LOG(ERROR) << errorMsg;
    }
  } else {
    errorMsg = c10::str(
        logPrefix(),
        "Future for ",
        futDescription,
        " timed out after ",
        timeOutMilSec.count(),
        " ms");
    LOG(ERROR) << errorMsg;
  }
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
}

c10::intrusive_ptr<c10d::Backend> ProcessGroupCNCL::createProcessGroupCNCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::milliseconds& timeout) {
  auto options = Options::create();
  options->timeout = timeout;
  return c10::make_intrusive<ProcessGroupCNCL>(store, rank, size, options);
}

} // namespace torch_mlu
