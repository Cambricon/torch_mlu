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

#include <torch/csrc/distributed/c10d/TraceUtils.h>

#include <map>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include "cncl_utils.h"
#include "framework/core/stream_guard.h"
#include "aten/utils/utils.h"

#ifdef TEST_COVERAGE
extern "C" void __gcov_flush();
#endif

namespace torch_mlu {

constexpr const char* const k_cncl_aborted_comm_store_key = "CNCLABORTEDCOMM";

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

void CNCLComm::cnclCommAbort(c10::optional<std::string> comm_failure_reason) {
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

// RAII helper class to manage CNCL group API and CUDA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct AutoCnclGroup {
  AutoCnclGroup() {
    // TODO(zhiguangda): using lock in order to follow CUDA,
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
    // average operation is done in pre or post process
    {c10d::ReduceOp::AVG, cnclSum},
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
    if (input.scalar_type() == at::kBool) {
      if (reduce_op == c10d::ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see cnclDataType mapping).
        return cnclMax;
      }
      if (reduce_op == c10d::ReduceOp::AVG) {
        C10_THROW_ERROR(
            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
      }
    }
    return cncl_op.at(reduce_op);
  } catch (const std::out_of_range& e) {
    switch (reduce_op) {
      case c10d::ReduceOp::BAND:
        throw std::runtime_error("Cannot use ReduceOp.BAND with CNCL");
        break;
      case c10d::ReduceOp::BOR:
        throw std::runtime_error("Cannot use ReduceOp.BOR with CNCL");
        break;
      case c10d::ReduceOp::BXOR:
        throw std::runtime_error("Cannot use ReduceOp.BXOR with CNCL");
        break;
      default:
        throw std::runtime_error("Unhandled ReduceOp");
        break;
    }
  }
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  return std::to_string(devices[0].index());
}

std::string getKeySendRecv(int my_rank, int peer) {
  int low_rank = my_rank < peer ? my_rank : peer;
  int high_rank = my_rank < peer ? peer : my_rank;
  std::string send_recv_pair =
      std::to_string(low_rank) + ":" + std::to_string(high_rank);
  return send_recv_pair;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    // tensors must all be on the same device, or all on distinct devices.
    // The line below assumes that constraint has already been enforced
    // (by check_gpu_tensors_same_device or
    // check_mlu_tensors_different_devices).
    if (res.size() == 0 || tensor.device() != res[0]) {
      res.push_back(tensor.device());
    }
  }
  return res;
}

// [Sync Streams] Helper that lets the input cncl_stream to wait for the current
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
void syncStreams(
    const std::vector<at::Device>& devices,
    std::vector<torch_mlu::MLUEvent>& cncl_events,
    std::vector<torch_mlu::MLUStream>& cncl_streams) {
  for (const auto i : c10::irange(devices.size())) {
    auto current_stream = torch_mlu::getCurrentMLUStream(devices[i].index());
    torch_mlu::MLUStream& cncl_stream = cncl_streams[i];
    torch_mlu::MLUEvent& cncl_event = cncl_events[i];
    cncl_event.place(current_stream);
    cncl_event.wait(cncl_stream);
  }
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
bool checkTensorsNeedFlatten(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor> other,
    int64_t world_size) {
  // The check below is moved form flatten_for_scatter_gather, if this func
  // is not needed any more, these checks need be moved back !!!
  if (tensor_lists.size() != 1 || other.size() != 1) {
    throw std::runtime_error(
        "MLU Tensors must be on a single MLU device per process");
  }

  if (tensor_lists[0].size() == 0) {
    throw std::runtime_error("Received an empty list");
  }

  if (tensor_lists[0].size() != world_size) {
    throw std::runtime_error(
        "Tensor list input to scatter/gather must match number of collective"
        " participants");
  }

  auto device = other[0].device();
  for (const auto& t : tensor_lists[0]) {
    if (t.numel() != other[0].numel()) {
      throw std::runtime_error(
          "All tensor operands to scatter/gather must have the same number of elements");
    }
    if (t.device() != device) {
      throw std::runtime_error("Expecting all tensors on the same device");
    }
  }

  auto input_base_ptr =
      torch_mlu::getMluTensorImpl(tensor_lists[0][0])->mlu_data_ptr();
  for (int i = 0; i < tensor_lists[0].size(); i++) {
    if (!tensor_lists[0][i].device().is_privateuseone() ||
        tensor_lists[0][i].is_sparse()) {
      return false;
    }
    // Check tensors contiguous
    if (!tensor_lists[0][i].is_contiguous(
            tensor_lists[0][i].suggest_memory_format())) {
      return false;
    }
    auto input_impl = torch_mlu::getMluTensorImpl(tensor_lists[0][i]);
    auto input_ptr = input_impl->mlu_data_ptr();
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

namespace cncl::detail {

template <typename Options>
at::Tensor procReduceInput(
    at::Tensor& input,
    Options& opts,
    bool canModifyInput,
    int world_size,
    bool avoidRecordStreams,
    c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work,
    torch_mlu::MLUStream& stream) {
  torch_mlu::mlu::MLUStreamGuard guard(stream);
  at::Tensor prep_input = input;
  if (opts.reduceOp == c10d::ReduceOp::AVG &&
      !at::isIntegralType(input.scalar_type(), true /*include bool*/)) {
    if (canModifyInput) {
      prep_input.div_(world_size);
    } else {
      // create new input tensor
      prep_input = at::native::div(input, world_size);
      if (avoidRecordStreams) {
        work->addStashedTesnor(prep_input);
      } else {
        torch_mlu::MLUCachingAllocator::recordStream(
            prep_input.storage().data_ptr(), stream);
      }
    }
  }
  return prep_input;
}
template <typename Options>
void procReduceOutput(
    at::Tensor& input,
    at::Tensor& output,
    Options opts,
    torch_mlu::MLUStream& stream,
    int world_size) {
  if (opts.reduceOp == c10d::ReduceOp::AVG &&
      at::isIntegralType(input.scalar_type(), false /*include bool*/)) {
    // Here we don't need to record output's stream because fn has
    // recorded
    torch_mlu::mlu::MLUStreamGuard guard(stream);
    output.floor_divide_(world_size);
  }
}

cnclResult_t reduce_scatter(
    at::Tensor& input,
    at::Tensor& output,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work,
    bool avoidRecordStreams,
    bool canModifyInput,
    int world_size,
    const c10d::ReduceScatterOptions& opts) {
  at::Tensor prep_input = procReduceInput(
      input,
      opts,
      canModifyInput,
      world_size,
      avoidRecordStreams,
      work,
      stream);

  auto input_impl = torch_mlu::getMluTensorImpl(prep_input);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto status = cnclReduceScatter(
      input_ptr,
      output_ptr,
      output.numel(),
      getCnclDataType(input.scalar_type()),
      getCnclReduceOp(opts.reduceOp, input),
      comm,
      stream.stream());

  procReduceOutput(input, output, opts, stream, world_size);
  return status;
}

cnclResult_t reduce(
    at::Tensor& input,
    at::Tensor& output,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work,
    bool avoidRecordStreams,
    bool canModifyInput,
    int world_size,
    const c10d::ReduceOptions& opts) {
  at::Tensor prep_input = procReduceInput(
      input,
      opts,
      canModifyInput,
      world_size,
      avoidRecordStreams,
      work,
      stream);
  const auto root = opts.rootRank + opts.rootTensor;
  const auto cncl_data_type = getCnclDataType(input.scalar_type());
  const auto cncl_reduce_op = getCnclReduceOp(opts.reduceOp, input);

  auto input_impl = torch_mlu::getMluTensorImpl(prep_input);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto status = cnclReduce(
      input_ptr,
      output_ptr,
      input.numel(),
      cncl_data_type,
      cncl_reduce_op,
      (int)root,
      comm,
      stream.stream());
  procReduceOutput(input, output, opts, stream, world_size);
  return status;
}

cnclResult_t allreduce(
    at::Tensor& input,
    at::Tensor& output,
    cnclComm_t comm,
    torch_mlu::MLUStream& stream,
    c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work,
    bool avoidRecordStreams,
    bool canModifyInput,
    int world_size,
    const c10d::AllreduceOptions& opts) {
  at::Tensor prep_input = procReduceInput(
      input,
      opts,
      canModifyInput,
      world_size,
      avoidRecordStreams,
      work,
      stream);
  auto input_impl = torch_mlu::getMluTensorImpl(input);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_impl = torch_mlu::getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto status = cnclAllReduce(
      input_ptr,
      output_ptr,
      input.numel(),
      getCnclDataType(input.scalar_type()),
      getCnclReduceOp(opts.reduceOp, input),
      comm,
      stream.stream());
  procReduceOutput(input, output, opts, stream, world_size);
  return status;
}

} // namespace cncl::detail

const int64_t ProcessGroupCNCL::k_watchdog_thread_sleep_millis = 10000;
const int64_t ProcessGroupCNCL::k_work_cleanup_thread_sleep_millis = 1000;
constexpr int64_t kWaitForAbortCommStoreKey = 1000;
constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupCNCL::cnclActiveGroupCounter_ = 0;

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
        work_cncl.op_timeout_.count(),
        ")");
  } else {
    workInfo = c10::str(
        "WorkCNCL(",
        "SeqNum=",
        work_cncl.seq_,
        ", c10d::OpType=",
        opTypeToString(work_cncl.opType_),
        ", Timeout(ms)=",
        work_cncl.op_timeout_.count(),
        ")");
  }
  return output << workInfo;
}

ProcessGroupCNCL::WorkCNCL::WorkCNCL(
    const std::vector<at::Device>& devices,
    int rank,
    c10d::OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs,
    bool desync_debug)
    : Work(rank, opType, profilingTitle, inputs),
      devices_(devices),
      work_start_time_(std::chrono::steady_clock::now()),
      seq_(seq) {
  // Creates the MLU event wrappers
  // Note: The actual events are lazily created when first recorded to.
  if (desync_debug) {
    cncl_start_events_ =
        std::make_shared<std::vector<torch_mlu::MLUEvent>>(devices.size());
  }
  cncl_end_events_ =
      std::make_shared<std::vector<torch_mlu::MLUEvent>>(devices.size());
  cncl_comms_.resize(devices.size());
}

ProcessGroupCNCL::WorkCNCL::~WorkCNCL() {}

// This is used by custom functions in cncl::detail
void ProcessGroupCNCL::WorkCNCL::addStashedTesnor(at::Tensor& t) {
  stashed_for_allocator_safety_->emplace_back(t);
};

bool ProcessGroupCNCL::WorkCNCL::isCompleted() {
  checkAndSetException();
  return exception() || finishedMLUExecutionInternal();
}

bool ProcessGroupCNCL::WorkCNCL::isSuccess() const {
  if (exception()) {
    // Already detected an exception
    return false;
  }

  return !checkForCNCLErrors(cncl_comms_) && finishedMLUExecutionInternal();
}

bool ProcessGroupCNCL::WorkCNCL::finishedMLUExecutionInternal() const {
  // Checking the work's corresponding MLU event's status
  try {
    for (const auto i : c10::irange(devices_.size())) {
      // Checking the work's corresponding CUDA events' status
      if (!(*cncl_end_events_)[i].query()) {
        return false;
      }
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

void ProcessGroupCNCL::WorkCNCL::synchronizeStreams() {
  for (const auto i : c10::irange(devices_.size())) {
    auto current_stream = getCurrentMLUStream(devices_[i].index());
    // Block the current stream on the CNCL stream
    (*cncl_end_events_)[i].wait(current_stream);
  }
  if (avoid_record_streams_) {
    stashed_for_allocator_safety_->clear();
  }
}

void ProcessGroupCNCL::WorkCNCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForCNCLErrors(cncl_comms_);
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
  synchronizeStreams();

  // In case of blocking, wait for the operation to complete.
  if (blocking_wait_) {
    // Wait for the operation to complete.
    while (!isCompleted()) {
      if (timedOut()) {
        // When operation times out due to some errors that are not
        // detected by cncl communicators, cnclCommWatchdog can not check this
        // time out error and thus can not abort cnclComms accordingly.
        // So explicitly abort cnclComms here before throwing this timed out
        // exception to users, after this, cnclCommWatchdog can detect cncl
        // communicators are aborted and clean up dev_cncl_comm_map_
        // accordingly.

        std::stringstream ss;
        ss << *this;
        auto timeout_error_msg =
            c10::str("Work ", ss.str(), " timed out in call to wait().");
        for (const auto& cncl_comm : cncl_comms_) {
          cncl_comm->cnclCommAbort(timeout_error_msg);
          const auto& store_key = getCnclAbortedCommStoreKey(
              buildCnclUniqueIdStr(cncl_comm->getCnclId()));
          auto rank_str = std::to_string(rank_);
          store_->set(
              store_key,
              std::vector<uint8_t>(
                  reinterpret_cast<const uint8_t*>(rank_str.data()),
                  reinterpret_cast<const uint8_t*>(rank_str.data()) +
                      rank_str.size()));
          LOG(INFO) << "[Rank " << rank_
                    << "] Wrote aborted communicator id to store: "
                    << store_key;
        }
        auto current_timepoint = std::chrono::steady_clock::now();
        auto timeElapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                current_timepoint - work_start_time_);
        std::string exception_msg = c10::str(
            "[Rank ",
            rank_,
            "] ",
            "Caught collective operation timeout: ",
            (*this),
            " ran for ",
            timeElapsed.count(),
            " milliseconds before timing out.");
        TORCH_CHECK(false, exception_msg);
      }
      // Check for errors and throw appropriate exception.
      checkAndThrowException();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
    checkAndThrowException();
  }

  // Device synchronize only after we've completed timeout checks.
  if (!barrier_tensors_.empty()) {
    // If we use the work to do barrier, we should block here
    for (auto& device : devices_) {
      auto currentStream = torch_mlu::getCurrentMLUStream(device.index());
      TORCH_CNRT_CHECK(cnrtQueueSync(currentStream));
    }
  }
}

void ProcessGroupCNCL::WorkCNCL::checkAndThrowException() {
  // Set the appropriate exception if found.
  checkAndSetException();

  // Throw an exception, only if we have a valid exception.
  if (exception()) {
    std::rethrow_exception(exception());
  }
}

// Helper that checks if the CNCL kernels are completed on the GPUs
bool ProcessGroupCNCL::WorkCNCL::finishedMLUExecution() {
  checkAndSetException();
  return finishedMLUExecutionInternal();
}

bool ProcessGroupCNCL::WorkCNCL::startedMLUExecutionInternal() const {
  for (const auto i : c10::irange(devices_.size())) {
    // Checking the work's corresponding MLU events' status
    if (!(*cncl_start_events_)[i].query()) {
      return false;
    }
  }
  return true;
}

void ProcessGroupCNCL::WorkCNCL::handleCNCLGuard(
    ErrorHandlingMode async_error_handling) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (exception_) {
    auto exception_msg = c10::str(
        "Some CNCL operations have failed or timed out. Due to the ",
        "asynchronous nature of MLU kernels, subsequent MLU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << exception_msg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupCNCL.WorkCNCL.handleCNCLGuard");
    if (async_error_handling == TearDown) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

bool ProcessGroupCNCL::WorkCNCL::timedOut() {
  auto current_timepoint = std::chrono::steady_clock::now();
  return (
      std::chrono::duration_cast<std::chrono::milliseconds>(
          current_timepoint - work_start_time_) >= op_timeout_);
}

// Same as calling synchronize().
bool ProcessGroupCNCL::WorkCNCL::wait(std::chrono::milliseconds timeout) {
  synchronizeInternal(timeout);
  return true;
}
void ProcessGroupCNCL::WorkCNCL::abort() {
  TORCH_CHECK(false, "ProcessGroupCNCL::WorkCNCL::abort not implemented.");
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
      devices_(w.devices_),
      cncl_start_events_(w.cncl_start_events_),
      cncl_end_events_(w.cncl_end_events_),
      cncl_comms_(w.cncl_comms_),
      blocking_wait_(w.blocking_wait_),
      op_timeout_(w.op_timeout_),
      work_start_time_(w.work_start_time_),
      seq_(w.seq_),
      start_trace_updated_(w.start_trace_updated_),
      store_(w.store_) {
  exception_ = w.exception_;
}

ProcessGroupCNCL::ProcessGroupCNCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(options),
      trace_key_start_(c10d::getTraceStartKey("CNCL", rank)),
      trace_key_end_(c10d::getTraceEndKey("CNCL", rank)),
      terminate_process_group_(false) {
  this->localDeviceCount_ = torch_mlu::device_count();
  TORCH_CHECK(
      torch_mlu::device_count() != 0,
      "ProcessGroupCNCL is only supported with MLUs, no MLUs found!");

  blocking_wait_ = getCvarBool(CNCL_BLOCKING_WAIT, false);

  // TODO:[PYTORCH-12498] only get CNCL_ASYNC_ERROR_HANDLING
  if (0) {
    async_error_handling_ = static_cast<ErrorHandlingMode>(
        getCvarInt(CNCL_ASYNC_ERROR_HANDLING, 0));
  } else {
    async_error_handling_ = static_cast<ErrorHandlingMode>(
        c10d::parseEnvVarIntDefault(CNCL_ASYNC_ERROR_HANDLING[0].c_str(), 0));
  }

  desync_debug_ = getCvarBool(CNCL_DESYNC_DEBUG, false) ||
      (dist_debug_level_ >= c10d::DebugLevel::Detail);

  avoid_record_streams_ = getCvarBool(TORCH_CNCL_AVOID_RECORD_STREAMS, false);

  if (blocking_wait_) {
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

  if (async_error_handling_ != NoHandling) {
    work_cleanup_thread_ =
        std::thread(&ProcessGroupCNCL::workCleanupLoop, this);
  }

  // Init global rank
  globalRank();
}

ProcessGroupCNCL::~ProcessGroupCNCL() {
  terminate_process_group_.store(true);

  watchdog_cv_.notify_one();
  cncl_comm_watchdog_thread_.join();

  if (async_error_handling_ != NoHandling) {
    work_meta_list_cv_.notify_one();
    work_cleanup_thread_.join();
  }

  {
    // Abort all CNCL Communicators on Process Group Destruction
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : dev_cncl_comm_map_) {
      auto& cncl_comms = it.second;

      for (const auto& cncl_comm : cncl_comms) {
        cncl_comm->cnclCommAbort();
      }
    }
  }
// gcov can not save the coverage data of the code run by subprocess,
// so we flush the coverge data manually
#ifdef TEST_COVERAGE
  __gcov_flush();
#endif
}

void ProcessGroupCNCL::abortTimedOutCollectives(
    std::unordered_set<std::string>& aborted_comm_ids) {
  std::unique_lock<std::mutex> lock(work_meta_list_mutex_);
  for (auto& work : work_meta_list_) {
    work.checkAndSetException();
    // Aborting CNCL Communicators due to errors is already handled above.
    if (work.exception()) {
      continue;
    }

    // Check for Timeouts in the WorkCNCL Operations, and abort all
    // communicators accordingly.
    if (work.timedOut()) {
      auto current_timepoint = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          current_timepoint - work.work_start_time_);
      std::string exception_msg = c10::str(
          "[Rank ",
          rank_,
          "] ",
          "Watchdog caught collective operation timeout: ",
          work,
          " ran for ",
          timeElapsed.count(),
          " milliseconds before timing out.");
      if (desync_debug_) {
        exception_msg += retrieveDesyncReport(store_, "CNCL", rank_, size_);
      }
      LOG(ERROR) << exception_msg;
      std::exception_ptr exception_ptr =
          std::make_exception_ptr(std::runtime_error(exception_msg));
      work.setException(exception_ptr);
      for (const auto& cncl_comm : work.cncl_comms_) {
        cncl_comm->cnclCommAbort(exception_msg);
        aborted_comm_ids.emplace(buildCnclUniqueIdStr(cncl_comm->getCnclId()));
      }
    }
  }
}

void ProcessGroupCNCL::cnclCommWatchdog() {
  try {
    LOG(INFO) << "[Rank " << rank_ << "] CNCL watchdog thread started!";
    cnclCommWatchdogInternal();
    LOG(INFO) << "[Rank " << rank_
              << "] CNCL watchdog thread terminated normally";
  } catch (std::exception& e) {
    LOG(INFO) << "[Rank " << rank_
              << "] CNCL watchdog thread terminated with exception: "
              << e.what();
  } catch (...) {
    LOG(INFO) << "[Rank " << rank_
              << "] CNCL watchdog thread terminated with unknown exception";
  }
}

void ProcessGroupCNCL::cnclCommWatchdogInternal() {
  while (!terminate_process_group_.load()) {
    std::unordered_set<std::string> aborted_comm_ids;
    std::unordered_set<std::string> all_comm_ids;

    {
      // Loop through the cache of communicators for CNCL errors.
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& it : dev_cncl_comm_map_) {
        auto& cncl_comms = it.second;

        for (const auto& cncl_comm : cncl_comms) {
          all_comm_ids.emplace(buildCnclUniqueIdStr(cncl_comm->getCnclId()));
        }
        std::exception_ptr cncl_error_exception =
            checkForCNCLErrors(cncl_comms);
        if (cncl_error_exception) {
          auto exception_msg =
              getExceptionMsgFromExceptionPtr(cncl_error_exception);
          LOG(INFO)
              << "[Rank " << rank_
              << "] Received CNCL errors for communicators in the cache: \n"
              << "CNCL error: \n"
              << exception_msg;

          if (blocking_wait_ || async_error_handling_ != NoHandling) {
            LOG(INFO) << "[Rank " << rank_
                      << "] Aborting communicators that received errors";
            // We abort CNCL communicators that have received errors from this
            // thread, and exceptions are set on the corresponding work objects.
            // The workCleanupThread will then loop through the unfinished
            // collectives and throw exceptions if an exception has been set on
            // any of the work objects from this thread.
            for (const auto& cncl_comm : cncl_comms) {
              // We are aborting remaining communicators due to an error in
              // at least one of these communicators, so propagate that reason
              // for better debugability.
              cncl_comm->cnclCommAbort(exception_msg);
              // Note that we don't remove the aborted communicators from the
              // cache. The reason is that if we do remove the communicator
              // from the cache, it is possible that a new collective operation
              // calls `cncl_commInitRank` to create a new communicator whereas
              // other ranks might have failed/timed out and didn't enter
              // `cncl_commInitRank`. As a result, when there is a failure on
              // a communicator the application receives an exception and its
              // their responsibility to destroy the process group and recreate
              // it to recover from errors.
              aborted_comm_ids.emplace(
                  buildCnclUniqueIdStr(cncl_comm->getCnclId()));
            }
          }
        }
      }
    }

    if (async_error_handling_ != NoHandling) {
      abortTimedOutCollectives(aborted_comm_ids);
    }

    if (blocking_wait_) {
      // When we abort a communicator on one rank, it is likely that might cause
      // other ranks to hang indefinitely. As a result, whenever we abort a
      // communicator, we write its ID to the store. The watchdog on other ranks
      // then monitor the store, find an aborted communicator ID and abort their
      // respective communicator as well.

      // Record the aborted communicators locally and in the store.
      for (const auto& aborted_comm_id : aborted_comm_ids) {
        aborted_comms_.emplace(aborted_comm_id);
        const auto& store_key = getCnclAbortedCommStoreKey(aborted_comm_id);
        auto rank_str = std::to_string(rank_);
        store_->set(
            store_key,
            std::vector<uint8_t>(
                reinterpret_cast<const uint8_t*>(rank_str.data()),
                reinterpret_cast<const uint8_t*>(rank_str.data()) +
                    rank_str.size()));
        LOG(INFO) << "[Rank " << rank_
                  << "] Watchdog wrote aborted communicator id to store: "
                  << store_key;
      }

      // Check for any communicators in the store and abort them if needed.
      for (const auto& comm_id : all_comm_ids) {
        if (aborted_comms_.find(comm_id) == aborted_comms_.end()) {
          // Check if we need to abort them if not already aborted (shouldn't
          // wait more than the watchdog sleep time.).
          const auto& store_key = getCnclAbortedCommStoreKey(comm_id);
          try {
            store_->wait(
                {store_key},
                std::chrono::milliseconds(kWaitForAbortCommStoreKey));
            auto val = store_->get(store_key);
            std::string rank(reinterpret_cast<char*>(val.data()), val.size());
            std::stringstream ss;
            ss << "[Rank " << rank_ << "] Found key in store: " << store_key
               << ", from rank: " << rank
               << ". This means that rank has aborted its CNCL communicators "
                  "previously and is not in a healthy state."
               << ". Aborting appropriate communicators";
            std::string abort_reason = ss.str();
            LOG(WARNING) << abort_reason;

            // Now abort the appropriate communicators.
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cncl_id_to_comm_map_.find(comm_id);
            TORCH_INTERNAL_ASSERT(it != cncl_id_to_comm_map_.end());
            for (const auto& cncl_comm : it->second) {
              // The reason we are aborting is because some other ranks have
              // aborted their communicators originally, so propagate that
              // reason.
              cncl_comm->cnclCommAbort(abort_reason);
            }
            aborted_comms_.emplace(comm_id);
            LOG(INFO) << "[Rank " << rank_
                      << "] Aborted communicators for key in store: "
                      << store_key;
          } catch (std::exception& e) {
            LOG(INFO) << "Did not find key in store: " << store_key
                      << ", error: " << e.what();
          }
        }
      }
    }

    std::unique_lock<std::mutex> lock(watchdog_cv_mutex_);
    watchdog_cv_.wait_for(
        lock,
        std::chrono::milliseconds(k_watchdog_thread_sleep_millis),
        [&]() -> bool { return terminate_process_group_.load(); });
  }
}

void ProcessGroupCNCL::workEnqueue(
    c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> work) {
  if (!terminate_process_group_.load()) {
    std::lock_guard<std::mutex> lock(work_meta_list_mutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    work_meta_list_.emplace_back(WorkCNCL(*work));
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
    std::vector<at::Device> devices,
    int rank,
    c10d::OpType opType,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputs) {
  return c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(
      devices, rank, opType, seq_, profilingTitle, inputs, desync_debug_);
}

void ProcessGroupCNCL::setSequenceNumberForGroup() {
} // CNCL just starts sequence numbers at 0.

uint64_t ProcessGroupCNCL::getSequenceNumberForGroup() {
  return seq_;
}

void ProcessGroupCNCL::workCleanupLoop() {
  bool done = false;
  while (!terminate_process_group_.load() || !done) {
    std::list<WorkCNCL> done_works;
    {
      std::unique_lock<std::mutex> lock(work_meta_list_mutex_);
      // We busy-poll the work vector every k_watchdog_thread_sleep_millis
      // milliseconds as long as the atomic is True.
      work_meta_list_cv_.wait_for(
          lock,
          std::chrono::milliseconds(k_work_cleanup_thread_sleep_millis),
          [&]() -> bool { return terminate_process_group_.load(); });

      for (auto it = work_meta_list_.begin(); it != work_meta_list_.end();
           /* no increment*/) {
        auto& work = *it;

        if (desync_debug_ && !work.exception()) {
          if (!work.start_trace_updated_ && work.isStarted() &&
              !terminate_process_group_.load() && !store_error_) {
            work.start_trace_updated_ = true;
            store_error_ = !c10d::traceUpdate(
                store_,
                trace_key_start_,
                work.seq_,
                opTypeToString(work.opType_));
          }
        }

        if (work.isCompleted()) {
          if (desync_debug_ && !work.exception()) {
            // To close the window between the check of work.isStarted() and
            // the check of work.isCompleted().
            if (!work.start_trace_updated_ &&
                !terminate_process_group_.load() && !store_error_) {
              store_error_ = !c10d::traceUpdate(
                  store_,
                  trace_key_start_,
                  work.seq_,
                  opTypeToString(work.opType_));
            }
            if (!terminate_process_group_.load() && !store_error_) {
              store_error_ = !c10d::traceUpdate(
                  store_,
                  trace_key_end_,
                  work.seq_,
                  opTypeToString(work.opType_));
            }
          }
          // Handle Exceptions on failed MLU operations and remove completed
          // workCNCL objects from work vector.
          if (!terminate_process_group_.load()) {
            work.handleCNCLGuard(async_error_handling_);
          }
          done_works.push_back(std::move(*it));
          it = work_meta_list_.erase(it);
        } else {
          // Increment the iterator if the current WorkCNCL object is not
          // completed.
          ++it;
        }
      }
      done = work_meta_list_.empty();
    }
    done_works.clear();
  }
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

std::vector<std::shared_ptr<CNCLComm>>& ProcessGroupCNCL::getCNCLComm(
    const std::string& devices_key,
    const std::vector<at::Device>& devices,
    c10d::OpType op_type,
    const int p2p_rank,
    const bool is_send_recv_self) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
        "Not able to create/get the CNCL Communicator since "
        "the MLU devices are not known");
  }

  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (dev_cncl_comm_map_.find(devices_key) != dev_cncl_comm_map_.end()) {
      // Reuse the cached communicator if there is one.
      return dev_cncl_comm_map_[devices_key];
    }
  }

  // CNCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<CNCLComm>> cncl_comms;
  cncl_comms.resize(devices.size());

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
    broadcastCNCLCliqueID(&clique_id, single_p2p_op, devices_key, p2p_rank);
  }

  std::vector<torch_mlu::MLUStream> stream_val;
  stream_val.reserve(devices.size());

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

  for (const auto i : c10::irange(devices.size())) {
    // world size and rank
    int num_ranks, rank_id;

    if (!single_p2p_op) {
      // Collective, all-to-all, or batch P2P
      // One rank for each device
      num_ranks = getSize() * devices.size();
      rank_id = getRank() * devices.size() + i;
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

    // Create the CNCL communicators for each MLU
    cncl_comms[i] =
        CNCLComm::create(num_ranks, rank_id, devices[i].index(), &clique_id);

    // Create streams
    stream_val.push_back(torch_mlu::getStreamFromPool(
        false /*options_->is_high_priority_stream*/, devices[i].index()));
  }

  // [Note 2 ]
  C10D_CNCL_CHECK(cnclGroupEnd(), c10::nullopt);

  // See [Group Start/End Note]
  for (size_t i = 0; i < cnclActiveGroupCounter_; ++i) {
    C10D_CNCL_CHECK(cnclGroupStart(), c10::nullopt);
  }

  cncl_streams_.emplace(devices_key, std::move(stream_val));
  updateCnclStream(&clique_id, cncl_streams_.at(devices_key).front());

  cncl_events_.emplace(
      std::piecewise_construct,
      std::make_tuple(devices_key),
      std::make_tuple(devices.size()));

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(mutex_);

  // Record the communicators based on cnclCliqueId.
  cncl_id_to_comm_map_.emplace(buildCnclUniqueIdStr(clique_id), cncl_comms);

  // Move the CNCL resource to cache
  dev_cncl_comm_map_.emplace(devices_key, std::move(cncl_comms));
  return dev_cncl_comm_map_[devices_key];
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
          "you need to check if the current device is corrent when calling the getCnclComm. " +
          "If it's wrong, it might have introduced an error.";
      TORCH_WARN(warning);
    }
  });
  std::vector<at::Device> devices = {device};
  const auto key = getKeyFromDevices(devices);
  auto& cnclComms = getCNCLComm(key, devices, c10d::OpType::UNKNOWN);
  TORCH_CHECK(
      cnclComms.size() == 1,
      "Except cnclComms.size() == 1, but cnclComms.size() == ",
      cnclComms.size());
  auto ret_cnclComm = cnclComms[0]->getCnclComm();
  int64_t cncl_comm =
      static_cast<int64_t>(reinterpret_cast<intptr_t>(ret_cnclComm));
  return cncl_comm;
}

std::exception_ptr ProcessGroupCNCL::WorkCNCL::checkForCNCLErrors(
    const std::vector<std::shared_ptr<CNCLComm>>& cncl_comms) const {
  return checkForCNCLErrorsInternal(cncl_comms);
}

std::exception_ptr ProcessGroupCNCL::checkForCNCLErrors(
    const std::vector<std::shared_ptr<CNCLComm>>& cncl_comms) {
  return checkForCNCLErrorsInternal(cncl_comms);
}

std::exception_ptr ProcessGroupCNCL::checkForCNCLErrorsInternal(
    const std::vector<std::shared_ptr<CNCLComm>>& cncl_comms) {
  for (const auto& cncl_comm : cncl_comms) {
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
  }

  return nullptr;
}
namespace {

// Check validity of tensor
void check_mlu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.device().is_privateuseone() || tensor.is_sparse()) {
    throw std::runtime_error("Tensors must be MLU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    throw std::runtime_error("Tensors must be contiguous");
  }
}

// Check that all `tensors' have the same type and shape and are distributed
// across distinct MLUs.
void check_mlu_tensors_different_devices(
    const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    TORCH_CHECK(false, "Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(torch_mlu::device_count())) {
    TORCH_CHECK(
        false,
        "Tensor list mustn't be larger than the number of available MLUs");
  }

  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (!t.device().is_privateuseone() || t.is_sparse()) {
      TORCH_CHECK(false, "Tensors must be MLU and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      TORCH_CHECK(false, "Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      TORCH_CHECK(false, "Tensors must have identical size");
    }
    if (t.strides() != first.strides()) {
      TORCH_CHECK(false, "Tensors must have identical strides");
    }
    if (!t.is_contiguous(t.suggest_memory_format())) {
      TORCH_CHECK(false, "Tensors must be contiguous");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      TORCH_CHECK(false, "Tensors must be on distinct MLU devices");
    }
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

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  // Only support one process one card.
  // This function create a tensor with no stride which is different from the
  // same name function with deviceIdx parameter.
  return {c10d::newLikeFlat(tensor_lists[0])};
}
} // namespace

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    c10d::OpType op_type,
    const char* profilingTitle) {
  seq_++;

  const auto devices = getDeviceList(inputs);
  const auto key = getKeyFromDevices(devices);

  auto& cncl_comms = getCNCLComm(key, devices, op_type);

  auto& cncl_streams = cncl_streams_[key];

  const bool inputs_same_dev = (devices.size() == 1);
  // First let CNCL stream wait for input tensors allocation stream
  syncStreams(devices, cncl_events_[key], cncl_streams);

  if (coalescing_active_) {
    coalescedDevices_.push_back(devices);
  }

  // Work itself will create the CNCL events on all MLUs of tensors
  bool can_profile = outputs.size() == 1;
  auto work = initWork(
      devices,
      rank_,
      op_type,
      can_profile ? profilingTitle : nullptr,
      can_profile ? c10::optional<std::vector<at::Tensor>>(inputs)
                  : c10::nullopt);

  // Store references to outputs to be used by WorkCNCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoid_record_streams_) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  torch_mlu::mlu::OptionalMLUGuard mlu_guard;

  // Start event should only be recorded before the cnclGroupStart()
  if (desync_debug_) {
    for (const auto i : c10::irange(devices.size())) {
      auto& cncl_stream = cncl_streams[i];
      (*work->cncl_start_events_)[i].place(cncl_stream);
    }
  }

  pre(cncl_streams, work);
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
    for (const auto i : c10::irange(inputs.size())) {
      if (!inputs_same_dev || (inputs_same_dev && i == 0)) {
        mlu_guard.set_index(devices[i].index());
      }
      decltype(i) stream_comm_i = (inputs_same_dev ? 0 : i);
      auto& cncl_stream = cncl_streams[stream_comm_i];
      auto& cncl_comm = cncl_comms[stream_comm_i];
      // Both `inputs' and `outputs' are created on a worker stream and used in
      // different cnclStreams.  Hence, both must record the cncl_stream to
      // prevent being freed before the collective finishes.
      //
      // We only record `inputs' here, and leave recording `outputs' to `fn' for
      // operations where `inputs' and `outputs' are not the same.
      if (!avoid_record_streams_) {
        torch_mlu::MLUCachingAllocator::recordStream(
            inputs[i].storage().data_ptr(), cncl_stream);
      }

      C10D_CNCL_CHECK(
          fn(inputs[i],
             outputs[i],
             cncl_comm->getCnclComm(),
             cncl_stream,
             work),
          cncl_comm->getCnclCommFailureReason());
    }
  }

  post(cncl_streams, work);

  // End event should only be recorded after the cnclGroupEnd()
  for (const auto i : c10::irange(devices.size())) {
    auto& cncl_stream = cncl_streams[i];
    if (!coalescing_active_) {
      (*work->cncl_end_events_)[i].place(cncl_stream);
    }
    work->cncl_comms_[i] = cncl_comms[i];
  }

  {
    // Set current stream to init future's event with cncl stream
    torch_mlu::mlu::MLUMultiStreamGuard stream_guard(cncl_streams);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
        work->recordFunctionEndCallback_();
      });
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  work->blocking_wait_ = blocking_wait_;
  work->op_timeout_ = options_->timeout;
  work->avoid_record_streams_ = avoid_record_streams_;
  work->store_ = store_;

  if (async_error_handling_ != NoHandling) {
    workEnqueue(work);
  }

  return work;
}

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    c10d::OpType op_type,
    const char* profilingTitle) {
  return collective(
      inputs,
      outputs,
      fn,
      [](std::vector<torch_mlu::MLUStream>&,
         c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
      [](std::vector<torch_mlu::MLUStream>&,
         c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
      op_type,
      profilingTitle);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts) {
  check_mlu_tensors_different_devices(tensors);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        return cncl::detail::allreduce(
            input,
            output,
            comm,
            stream,
            work,
            avoid_record_streams_,
            true, /* allreduce is a inplace op*/
            getSize(),
            opts);
      },
      c10d::OpType::ALLREDUCE,
      "cncl:all_reduce");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceCoalescedOptions& opts) {
  auto total_numel = check_mlu_tensors_same_device(tensors);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        return cncl::detail::allreduce(
            input,
            output,
            comm,
            stream,
            work,
            avoid_record_streams_,
            true, /* allreduce is a inplace op*/
            getSize(),
            opts);
      },
      c10d::OpType::COALESCED,
      "cncl:allreduce_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts) {
  check_mlu_tensors_different_devices(tensors);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        const int root = opts.rootRank * tensors.size() + opts.rootTensor;
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
  check_mlu_tensors_different_devices(tensors);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        return cncl::detail::reduce(
            input,
            output,
            comm,
            stream,
            work,
            avoid_record_streams_,
            true, /*reduce is inplace operation*/
            getSize(),
            opts);
      },
      c10d::OpType::REDUCE,
      "cncl:reduce");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::
    allgather_into_tensor_coalesced(
        std::vector<at::Tensor>& outputs,
        std::vector<at::Tensor>& inputs,
        const c10d::AllgatherOptions& opts) {
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
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
  check_mlu_tensors_different_devices(input_tensors);
  auto output_tensors_ = output_tensors.back();

  bool same_size = check_same_size(output_tensors_);
  if (same_size) {
    // check if origin output tensors are flattened
    bool flattened =
        checkTensorsNeedFlatten(output_tensors, input_tensors, size_);

    // Flatten a vector of tensors into a single, stacked tensor.
    std::vector<at::Tensor> output_flattened;
    if (!flattened) {
      output_flattened =
          flatten_for_scatter_gather(output_tensors, input_tensors, size_);
    } else {
      output_flattened = {output_tensors_[0]};
    }

    return collective(
        input_tensors,
        output_flattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            cnclComm_t comm,
            torch_mlu::MLUStream& stream,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
          auto input_impl = torch_mlu::getMluTensorImpl(input);
          auto input_ptr = input_impl->mlu_data_ptr();
          auto output_impl = torch_mlu::getMluTensorImpl(output);
          auto output_ptr = output_impl->mlu_data_ptr();
          if (!avoid_record_streams_) {
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
        [&](std::vector<torch_mlu::MLUStream>& cncl_streams,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
        [&](std::vector<torch_mlu::MLUStream>& cncl_streams,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
          // Copy the flattened output tensors to the outputs.
          for (const auto i : c10::irange(output_tensors.size())) {
            torch_mlu::mlu::MLUStreamGuard guard(cncl_streams[i]);
            for (const auto j : c10::irange(output_tensors[i].size())) {
              if (!avoid_record_streams_) {
                torch_mlu::MLUCachingAllocator::recordStream(
                    output_tensors[i][j].storage().data_ptr(), cncl_streams[i]);
              }

              if (!flattened)
                output_tensors[i][j].copy_(output_flattened[i][j], true);
            }
          }
        },
        c10d::OpType::ALLGATHER,
        "cncl:all_gather");
  } else {
    const auto num_devices = output_tensors.size();
    const auto num_reduces = output_tensors[0].size();
    std::vector<c10::intrusive_ptr<c10d::Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        outputs_multi_dev[j] = output_tensors[j][i];
        inputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? input_tensors[j]
                                           : outputs_multi_dev[j];
      }
      auto broadcastOpts = c10d::BroadcastOptions{
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work =
          _broadcast_oop(outputs_multi_dev, inputs_multi_dev, broadcastOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
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
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::BroadcastOptions& opts) {
  check_mlu_tensors_different_devices(outputTensors);
  check_mlu_tensors_different_devices(inputTensors);

  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    TORCH_CHECK(
        false,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        const auto root = opts.rootRank * inputTensors.size() + opts.rootTensor;
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
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    TORCH_CHECK(
        false,
        "output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        if (!avoid_record_streams_) {
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

// _reduce_oop exposes an out-of-place reduce from PGCNCL
// Custom collectives may be implemented by coalescing reduce operations
// One use-case is implementing a vector reduce_scatter (reduce_scatter_v)
// where inputs are reduced and scattered unevenly among participating ranks
// Since reduce_scatter provides an out-of-place API, a reduce_scatter_v
// semantic implemented inside pg_cncl.reduce_scatter also needs to support
// out-of-place, for which an out-of-place reduce is required to be added
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_reduce_oop(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::ReduceOptions& opts) {
  check_mlu_tensors_different_devices(outputTensors);
  check_mlu_tensors_different_devices(inputTensors);
  // @lint-ignore CLANGTIDY
  auto tensor = outputTensors.back();
  // @lint-ignore CLANGTIDY
  auto in_tensor = inputTensors.back();
  if (tensor.numel() != in_tensor.numel()) {
    TORCH_CHECK(
        false,
        "Tensor input and output of _reduce_oop must have the same number of elements ");
  }

  int dev_in_group{0};
  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        return cncl::detail::reduce(
            input,
            output,
            comm,
            stream,
            work,
            avoid_record_streams_,
            false, /*reduce oop is out of place reduce*/
            getSize(),
            opts);
      },
      c10d::OpType::REDUCE,
      "cncl:_reduce_oop");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
  check_mlu_tensors_different_devices(output_tensors);

  auto input_tensors_ = input_tensors.back();

  bool same_size = check_same_size(input_tensors_);
  if (same_size) {
    // check if origin input tensors are flattened
    bool flattened =
        checkTensorsNeedFlatten(input_tensors, output_tensors, size_);

    // For AVG and float type tensor
    if (opts.reduceOp == c10d::ReduceOp::AVG &&
        !at::isIntegralType(
            input_tensors_[0].scalar_type(), true /*include bool*/)) {
      flattened = false;
    }

    // Flatten a vector of tensors into a single, stacked tensor.
    std::vector<at::Tensor> input_flattened;
    if (!flattened) {
      input_flattened =
          flatten_for_scatter_gather(input_tensors, output_tensors, size_);
    } else {
      input_flattened = {input_tensors_[0]};
    }

    return collective(
        input_flattened,
        output_tensors,
        [&](at::Tensor& input,
            at::Tensor& output,
            cnclComm_t comm,
            torch_mlu::MLUStream& stream,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
          if (!avoid_record_streams_) {
            torch_mlu::MLUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          return cncl::detail::reduce_scatter(
              input,
              output,
              comm,
              stream,
              work,
              avoid_record_streams_,
              !flattened,
              getSize(),
              opts);
        },
        [&](std::vector<torch_mlu::MLUStream>& cncl_streams,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
          if (avoid_record_streams_) {
            // We only need to stash inputTensors.
            //  - inputFlattened is stashed onto
            //  work->stashed_for_allocator_safety_
            //    in collective().
            //  - User-facing outputTensors is stashed onto work->outputs_
            //  in collective(),
            //    and should also be held by the user until after waiting on
            //    work_.
            auto& v = work->stashed_for_allocator_safety_;
            for (const auto i : c10::irange(input_tensors.size())) {
              v->insert(
                  v->end(), input_tensors[i].begin(), input_tensors[i].end());
            }
          }
          // Copy the input tensors to the flattened inputs.
          for (const auto i : c10::irange(input_tensors.size())) {
            torch_mlu::mlu::MLUStreamGuard guard(cncl_streams[i]);
            for (const auto j : c10::irange(input_tensors[0].size())) {
              if (!avoid_record_streams_) {
                torch_mlu::MLUCachingAllocator::recordStream(
                    input_tensors[i][j].storage().data_ptr(), cncl_streams[i]);
              }
              if (!flattened)
                input_flattened[i][j].copy_(input_tensors[i][j], true);
            }
          }
        },
        [&](std::vector<torch_mlu::MLUStream>& cncl_streams,
            c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {},
        c10d::OpType::REDUCE_SCATTER,
        "cncl:reduce_scatter");
  } else {
    const auto num_devices = input_tensors.size();
    const auto num_reduces = input_tensors[0].size();
    std::vector<c10::intrusive_ptr<c10d::Work>> works;
    startCoalescing();
    for (const auto i : c10::irange(num_reduces)) {
      std::vector<at::Tensor> inputs_multi_dev(num_devices);
      std::vector<at::Tensor> outputs_multi_dev(num_devices);
      for (const auto j : c10::irange(num_devices)) {
        // @lint-ignore CLANGTIDY
        inputs_multi_dev[j] = input_tensors[j][i];
        outputs_multi_dev[j] =
            // @lint-ignore CLANGTIDY
            i == (rank_ * num_devices + j) ? output_tensors[j]
                                           : inputs_multi_dev[j];
      }
      auto reduceOpts = c10d::ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i / num_devices),
          static_cast<int64_t>(i % num_devices),
          opts.timeout};
      auto work = _reduce_oop(outputs_multi_dev, inputs_multi_dev, reduceOpts);
      works.push_back(work);
    }
    auto work = endCoalescing();
    return work;
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::
    reduce_scatter_tensor_coalesced(
        std::vector<at::Tensor>& outputs,
        std::vector<at::Tensor>& inputs,
        const c10d::ReduceScatterOptions& opts) {
  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        if (!avoid_record_streams_) {
          torch_mlu::MLUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return cncl::detail::reduce_scatter(
            input,
            output,
            comm,
            stream,
            work,
            avoid_record_streams_,
            false, /*can not modify input*/
            getSize(),
            opts);
      },
      c10d::OpType::COALESCED,
      "cncl:reduce_scatter_tensor_coalesced");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_reduce_scatter_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10d::ReduceScatterOptions& opts) {
  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(
        false, "input tensor must be the same type as the output tensor.");
  }

  if (input_tensor.numel() != output_tensor.numel() * size_) {
    TORCH_CHECK(
        false,
        "input tensor must be the same size as output size times world size.");
  }

  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        if (!avoid_record_streams_) {
          torch_mlu::MLUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
        }
        return cncl::detail::reduce_scatter(
            input,
            output,
            comm,
            stream,
            work,
            avoid_record_streams_,
            false, /*input not copyed can not be modified*/
            getSize(),
            opts);
      },
      c10d::OpType::_REDUCE_SCATTER_BASE,
      "cncl:reduce_scatter");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::barrier(
    const c10d::BarrierOptions& opts) {
  std::vector<at::Device> devices;
  // Device to use for barrier.
  int barDevIdx = -1;

  // Select device to use for barrier
  // 1st choice: User defined device ids if provided.
  if (!opts.device_ids.empty()) {
    barDevIdx = opts.device_ids[0];
    devices.push_back(at::Device(at::DeviceType::PrivateUse1, barDevIdx));
    // No 2nd choice for PT2.1 since the c10::Backend has no getBoundDeviceId.
  } else if (!usedDeviceIdxs_.empty()) {
    // 3rd choice: infer device id from the used device ids.
    barDevIdx = *usedDeviceIdxs_.begin();
    devices.push_back(at::Device(at::DeviceType::PrivateUse1, barDevIdx));
  } else if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a CNCL collective being called
    // Here we have to use the best guesses and will use a single MLU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different MLU
    // Note: it is better to use global rank because the group-local rank can be
    // offset wrt the device id if intra-node MLUs are sharded into multiple
    // dimensions.
    barDevIdx = static_cast<int16_t>(globalRank() % localDeviceCount_);
    LOG(WARNING) << c10::str(
        "Rank ",
        this->getRank(),
        " using MLU ",
        barDevIdx,
        " to perform barrier as devices used by this process are currently unknown. ",
        "This can potentially cause a hang if this rank to MLU mapping is incorrect.",
        "Specify device_ids in barrier() to force use of a particular device.");
    devices.push_back(at::Device(at::DeviceType::PrivateUse1, barDevIdx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.push_back(at::Device(at::DeviceType::PrivateUse1, usedDeviceIdx));
    }
  }

  TORCH_CHECK_WITH(
      ValueError,
      barDevIdx >= 0,
      "Failed to infer a MLU device id to perform barrier. ");

  // Create a dummy tensor on the device
  // Note: we use zeros() instead of empty() to prevent barrier from triggering
  // alarm when NaN checker is enabled.
  std::vector<at::Tensor> barrierTensors;
  barrierTensors.reserve(localDeviceCount_);
  torch_mlu::mlu::OptionalMLUGuard mlu_guard;
  for (auto& device : devices) {
    mlu_guard.set_index(device.index());
    barrierTensors.push_back(at::zeros(
        {1},
        at::TensorOptions()
            .device(at::DeviceType::PrivateUse1)
            .dtype(at::kFloat)));
  }

  // All reduce to achieve the barrier
  auto work = allreduce(barrierTensors);

  // Work will take over barrierTensors
  auto cncl_work = dynamic_cast<ProcessGroupCNCL::WorkCNCL*>(work.get());
  TORCH_CHECK(cncl_work);
  cncl_work->barrier_tensors_ = std::move(barrierTensors);

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
  std::vector<at::Tensor> input_tensors = {input_tensor};
  std::vector<at::Tensor> output_tensors = {output_tensor};

  return collective(
      input_tensors,
      output_tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        auto dtype = getCnclDataType(input.scalar_type());

        // When equal split, use cnclAlltoAll to improve performance
        if (output_split_sizes.size() == 0 && input_split_sizes.size() == 0) {
          int64_t cnt = input.numel() / size_;
          if (cnt == 0) {
            return CNCL_RET_SUCCESS;
          }
          if (!avoid_record_streams_) {
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

          size_t input_elem_size = getCnnlTypeSize(
              dynamic_cast<MLUTensorImpl*>(input_impl->external_.get())
                  ->cnnl_data_type_);

          if (!avoid_record_streams_) {
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
  TORCH_MLU_CHECK(
      input_tensors.size() == size_ && output_tensors.size() == size_,
      "Size of input tensor list not equal group size");

  auto device = output_tensors[0].device();
  for (size_t r = 0; r < output_tensors.size(); r++) {
    check_mlu_single_tensor(output_tensors[r]);
    check_mlu_single_tensor(input_tensors[r]);
    TORCH_MLU_CHECK(
        device == output_tensors[r].device() &&
            device == input_tensors[r].device(),
        "Tensors must be on the same device");
  }

  std::vector<at::Tensor> input_tensor0 = {input_tensors[0]};
  std::vector<at::Tensor> output_tensor0 = {output_tensors[0]};

  return collective(
      input_tensor0,
      output_tensor0,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        torch_mlu::cncl::detail::all2all(
            output_tensors, input_tensors, comm, stream);
        return CNCL_RET_SUCCESS;
      },
      [&](std::vector<torch_mlu::MLUStream>& cncl_streams,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        if (avoid_record_streams_) {
          // input_tensor0 and output_tensor0 are stashed redundantly by
          // collective(), but that's ok.
          auto& v = work->stashed_for_allocator_safety_;
          v->insert(v->end(), input_tensors.begin(), input_tensors.end());
          v->insert(v->end(), output_tensors.begin(), output_tensors.end());
        }
      },
      [&](std::vector<torch_mlu::MLUStream>& cncl_streams,
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
  check_mlu_tensors_different_devices(inputTensors);
  c10d::assertSingleElementInput(invalidArgument, inputTensors);

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

    const auto& options = inputTensors[0].options();
    const auto& sizes = inputTensors[0].sizes();
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
      inputTensors,
      outputs,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoid_record_streams_) {
            for (auto output : outputs) {
              torch_mlu::MLUCachingAllocator::recordStream(
                  output.storage().data_ptr(), stream);
            }
          }
        }
        torch_mlu::cncl::detail::gather(
            inputTensors[0], outputs, comm, stream, root);
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

  // TODO:[PYTORCH-11352]
  return collective(
      outputTensors,
      inputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          if (!avoid_record_streams_) {
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
      "cncl:scatter");
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
    std::vector<at::Tensor>& tensors,
    Fn fn,
    int peer,
    c10d::OpType op_type,
    const char* profilingTitle) {
  seq_++;
  // avoid_record_streams_ note:
  // send, recv, and irecv should be ok with avoid_record_streams,
  // However, for isend, I don't think the API requires the user
  // to wait() on the returned handle, so ProcessGroupCNCL can't know
  // when it's safe to release the input back to the allocator,
  // and the present call has no way to know it's not an isend.
  // Therefore, we warn and fall back to the typical recordStream logic:
  TORCH_WARN_ONCE(
      avoid_record_streams_,
      "CNCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point "
      "collectives.");

  std::string key;
  const auto devices = getDeviceList(tensors);
  const bool is_p2p_op = c10d::isP2POp(op_type);

  int p2p_rank = 0, p2p_target_rank = 0;
  bool is_send_recv_self = false;

  // For batch_isend_irecv, cnclGroupStart() would be called upfront
  bool batchP2P = cnclActiveGroupCounter_ > 0;

  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevices(devices);
    p2p_rank = rank_;
    p2p_target_rank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2p_rank = rank_ <= peer ? 0 : 1;
    is_send_recv_self = rank_ == peer;
    p2p_target_rank = is_send_recv_self ? 0 : 1 - p2p_rank;
  }

  auto& cncl_comms =
      getCNCLComm(key, devices, op_type, p2p_rank, is_send_recv_self);

  if (coalescing_active_) {
    coalescedDevices_.push_back(devices);
  }

  auto& cncl_streams = cncl_streams_[key];
  syncStreams(devices, cncl_events_[key], cncl_streams);

  // Work itself will create the CNCL events on all MLUs of tensors
  bool can_profile = tensors.size() == 1;
  auto work = initWork(
      devices,
      rank_,
      op_type,
      can_profile ? profilingTitle : nullptr,
      can_profile ? c10::optional<std::vector<at::Tensor>>(tensors)
                  : c10::nullopt);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(tensors);

  torch_mlu::mlu::OptionalMLUGuard mlu_guard;

  // Start event should only be recorded before the cnclGroupStart()
  if (desync_debug_) {
    for (const auto i : c10::irange(tensors.size())) {
      auto& cncl_stream = cncl_streams[i];
      (*work->cncl_start_events_)[i].place(cncl_stream);
    }
  }

  for (const auto i : c10::irange(tensors.size())) {
    mlu_guard.set_index(devices[i].index());
    auto& cncl_stream = cncl_streams[i];
    torch_mlu::MLUCachingAllocator::recordStream(
        tensors[i].storage().data_ptr(), cncl_stream);
  }

  {
    AutoCnclGroup cncl_group_guard;
    for (const auto i : c10::irange(tensors.size())) {
      torch_mlu::mlu::MLUGuard guard(devices[i]);
      auto& cncl_stream = cncl_streams[i];
      C10D_CNCL_CHECK(
          fn(tensors[i],
             cncl_comms[i]->getCnclComm(),
             cncl_stream,
             p2p_target_rank),
          cncl_comms[i]->getCnclCommFailureReason());
    }
  }

  // End event should only be recorded after the cnclGroupEnd()
  for (const auto i : c10::irange(devices.size())) {
    auto& cncl_stream = cncl_streams_[key][i];
    if (!coalescing_active_) {
      (*work->cncl_end_events_)[i].place(cncl_stream);
    }
    work->cncl_comms_[i] = cncl_comms[i];
  }

  work->blocking_wait_ = blocking_wait_;
  work->op_timeout_ = options_->timeout;
  work->store_ = store_;

  {
    // Set current stream to init future's event with cncl stream
    torch_mlu::mlu::MLUMultiStreamGuard stream_guard(cncl_streams);
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

  return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::send(
    std::vector<at::Tensor>& tensors,
    int dst_rank,
    int /* unused */) {
  check_mlu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& input,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          int dst) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        return cnclSend(
            input_impl->mlu_data_ptr(),
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
  check_mlu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::MLUStream& stream,
          int src) {
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        return cnclRecv(
            output_impl->mlu_data_ptr(),
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

const int& ProcessGroupCNCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
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
  coalescedDevices_.clear();
  coalescing_active_ = true;
  groupStart();
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::endCoalescing() {
  groupEnd();

  coalescing_active_ = false;

  if (coalescedDevices_.size() == 0) {
    return nullptr;
  }

  // `coalescedDevices_` should have same set of devices across collectives
  auto devices = coalescedDevices_[0];

  // Create Work object
  auto work = initWork(
      devices, rank_, c10d::OpType::COALESCED, "cncl:coalesced", c10::nullopt);

  const auto key = getKeyFromDevices(devices);
  auto& cncl_streams = cncl_streams_[key];

  for (const auto i : c10::irange(devices.size())) {
    (*work->cncl_end_events_)[i].place(cncl_streams[i]);
  }

  work->blocking_wait_ = blocking_wait_;
  work->avoid_record_streams_ = avoid_record_streams_;

  if (avoid_record_streams_) {
    // other functions expect an initialized ptr if avoid_record_streams_ is set
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>();
  }

  return work;
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
