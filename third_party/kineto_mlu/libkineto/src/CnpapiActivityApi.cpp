// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CnpapiActivityApi.h"

#include <assert.h>
#include <chrono>

#include "Logger.h"
#include "cnpapi_strings.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

// TODO: do we want this to be configurable?
// Set to 2MB to avoid constantly creating buffers (espeically for networks
// that has many small memcpy such as sparseNN)
// Consider putting this on huge pages?
constexpr size_t kBufSize(2 * 1024 * 1024);

CnpapiActivityApi& CnpapiActivityApi::singleton() {
  static CnpapiActivityApi instance;
  return instance;
}

void CnpapiActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_CNPAPI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  VLOG(2) << "pushCorrelationID(" << id << ")";
  switch(type) {
    case Default:
      CNPAPI_CALL(cnpapiActivityPushExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, id));
        break;
    case User:
      CNPAPI_CALL(cnpapiActivityPushExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, id));
  }
#endif
}

void CnpapiActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_CNPAPI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch(type) {
    case Default:
      CNPAPI_CALL(cnpapiActivityPopExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, nullptr));
        break;
    case User:
      CNPAPI_CALL(cnpapiActivityPopExternalCorrelationId(
        CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, nullptr));
  }
#endif
}

static bool nextActivityRecord(
    uint64_t* buffer,
    size_t valid_size,
    cnpapiActivity*& record) {
#ifdef HAS_CNPAPI
  cnpapiResult status = CNPAPI_CALL_NOWARN(
      cnpapiActivityGetNextRecord(buffer, valid_size, &record));
  if (status != CNPAPI_SUCCESS) {
    if (status != CNPAPI_ERROR_MAX_LIMIT_REACHED) {
      CNPAPI_CALL(status);
    }
    record = nullptr;
  }
#endif
  return record != nullptr;
}

void CnpapiActivityApi::setMaxBufferSize(int size) {
  maxMluBufferCount_ = 1 + size / kBufSize;
}

#ifdef HAS_CNPAPI
void CnpapiActivityApi::bufferRequestedTrampoline(
    uint64_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  singleton().bufferRequested(buffer, size, maxNumRecords);
}

void CnpapiActivityApi::bufferRequested(
    uint64_t** buffer, size_t* size, size_t* maxNumRecords) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedMluTraceBuffers_.size() >= maxMluBufferCount_) {
    stopCollection = true;
    LOG(WARNING) << "Exceeded max MLU buffer count ("
                 << allocatedMluTraceBuffers_.size()
                 << " > " << maxMluBufferCount_
                 << ") - terminating tracing";
  }

  auto buf = std::make_unique<CnpapiActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedMluTraceBuffers_[*buffer] = std::move(buf);

  *maxNumRecords = 0;
}
#endif

std::unique_ptr<CnpapiRecordSet>
CnpapiActivityApi::activityBuffers() {
#ifdef HAS_CNPAPI
  VLOG(1) << "Flushing MLU activity buffers";
  time_point<system_clock> t1;
  if (VLOG_IS_ON(1)) {
    t1 = system_clock::now();
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libcnpapi and mutex_ is acquired there.
  CNPAPI_CALL(cnpapiActivityFlushAll());
  if (VLOG_IS_ON(1)) {
    flushOverhead =
        duration_cast<microseconds>(system_clock::now() - t1).count();
  }
#endif
  std::lock_guard<std::mutex> guard(mutex_);
  // Transfer ownership of buffers to caller. A new map is created on-demand.
  return std::move(all_records_);
}

#ifdef HAS_CNPAPI
int CnpapiActivityApi::processActivitiesForBuffer(
    uint64_t* buf,
    size_t validSize) {
  int count = 0;
  if (buf && validSize) {
    cnpapiActivity* raw_record{nullptr};
    while ((nextActivityRecord(buf, validSize, raw_record))) {
      processRecords(raw_record);
      ++count;
    }
  }
  return count;
}
void CnpapiActivityApi::processRecords(cnpapiActivity* raw_record) {
  switch (raw_record->type) {
    case CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION: {
      auto correlation =
        reinterpret_cast<cnpapiActivityExternalCorrelation*>(raw_record);
      all_records_->external_correlation_records.emplace_back(
        correlation->external_type,
        correlation->external_id,
        correlation->correlation_id);
      break;
    }
    case CNPAPI_ACTIVITY_TYPE_CNNL_API:
    case CNPAPI_ACTIVITY_TYPE_CNNL_EXTRA_API:
    case CNPAPI_ACTIVITY_TYPE_CNDRV_API:
    case CNPAPI_ACTIVITY_TYPE_CNCL_API:
    case CNPAPI_ACTIVITY_TYPE_CNRT_API: {
      auto runtime = reinterpret_cast<cnpapiActivityAPI*>(raw_record);
      all_records_->runtime_records.emplace_back(runtime->type,
        runtime->correlation_id,
        runtime->cbid, runtime->start, runtime->end,
        runtime->process_id, runtime->thread_id);
      break;
    }
    case CNPAPI_ACTIVITY_TYPE_KERNEL: {
      auto kernel = reinterpret_cast<cnpapiActivityKernel*>(raw_record);
      all_records_->kernel_records.emplace_back(kernel->correlation_id,
        kernel->start, kernel->end, kernel->queued,
        kernel->device_id, kernel->name, kernel->queue_id,
        kernel->kernel_type, kernel->dimx, kernel->dimy,
        kernel->dimz, kernel->context_id,
        kernel->tasktopo_id, kernel->tasktopo_node_id);
      break;
    }
    case CNPAPI_ACTIVITY_TYPE_MEMCPY: {
      auto memcpy = reinterpret_cast<cnpapiActivityMemcpy*>(raw_record);
      all_records_->memcpy_records.emplace_back(memcpy->correlation_id,
        memcpy->bytes, memcpy->copy_type, memcpy->start,
        memcpy->end, memcpy->device_id,
        memcpy->queue_id,
        memcpy->context_id);
      break;
    }
    case CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP: {
      auto memcpyp2p = reinterpret_cast<cnpapiActivityMemcpyPtoP*>(raw_record);
      all_records_->memcpy_p2p_records.emplace_back(memcpyp2p->correlation_id,
        memcpyp2p->bytes, memcpyp2p->copy_type,
        memcpyp2p->start, memcpyp2p->end, memcpyp2p->device_id,
        memcpyp2p->src_device_id, memcpyp2p->dst_device_id,
        memcpyp2p->queue_id, memcpyp2p->context_id);
      break;
    }
    case CNPAPI_ACTIVITY_TYPE_MEMSET: {
      auto memset = reinterpret_cast<cnpapiActivityMemset*>(raw_record);
      all_records_->memset_records.emplace_back(memset->correlation_id,
        memset->bytes, memset->start, memset->end,
        memset->device_id, memset->queue_id, memset->context_id);
      break;
    }
    case CNPAPI_ACTIVITY_TYPE_OVERHEAD: {
      auto overhead = reinterpret_cast<cnpapiActivityOverhead*>(raw_record);
      all_records_->overhead_records.emplace_back(overhead->overhead_type,
        overhead->start, overhead->end);
      break;
    }
    case CNPAPI_ACTIVITY_TYPE_ATOMIC_OPERATION: {
      auto atomic = reinterpret_cast<cnpapiActivityAtomicOperation*>(raw_record);
      AtomicFlagType flag;
      if (atomic->operation_type == CNPAPI_ACTIVITY_ATOMIC_OP_REQUEST) {
        flag = atomic->req_flag;
      } else if (atomic->operation_type == CNPAPI_ACTIVITY_ATOMIC_OP_COMPARE) {
        flag = atomic->com_flag;
      }
      all_records_->atomic_op_records.emplace_back(atomic->correlation_id,
        atomic->start, atomic->end, atomic->queued, atomic->device_id,
        atomic->queue_id, atomic->context_id, atomic->operation_type,
        flag, atomic->value);
      break;
    }
    default:
      LOG(WARNING) << "Unexpected activity type: " << raw_record->type;
      break;
  }
}
#endif

void CnpapiActivityApi::clearActivities() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedMluTraceBuffers_.empty()) {
      return;
    }
  }
  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libcnpapi and mutex_ is acquired there.
#ifdef HAS_CNPAPI
  CNPAPI_CALL(cnpapiActivityFlushAll());
#endif
  // FIXME: We might want to make sure we reuse
  // the same memory during warmup and tracing.
  // Also, try to use the amount of memory required
  // for active tracing during warmup.
  std::lock_guard<std::mutex> guard(mutex_);
  // Throw away ready buffers as a result of above flush
  all_records_ = nullptr;
}

#ifdef HAS_CNPAPI
void CnpapiActivityApi::bufferCompletedTrampoline(
    uint64_t* buffer,
    size_t size,
    size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void CnpapiActivityApi::bufferCompleted(
    uint64_t* buffer,
    size_t size,
    size_t validSize) {

  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedMluTraceBuffers_.find(buffer);
  if (it == allocatedMluTraceBuffers_.end()) {
    LOG(ERROR) << "bufferCompleted called with unknown buffer: "
               << (void*) buffer;
    return;
  }

  processActivitiesForBuffer(buffer, validSize);
  allocatedMluTraceBuffers_.erase(it);
}

namespace {

const std::vector<cnpapi_CallbackIdCNRT> enabledCnrtCbidList = {
  CNPAPI_CNRT_TRACE_CBID_cnrtMalloc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMallocBatch,
  CNPAPI_CNRT_TRACE_CBID_cnrtMallocHost,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatch,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatchByDesc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatchByDescArray,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyByDesc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyByDescArray,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemset,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyPeer,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD8,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD32,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD8Async,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD32Async,
  CNPAPI_CNRT_TRACE_CBID_cnrtSyncDevice,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyPeerAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtHostMalloc,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueCreate,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueDestroy,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueQuery,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueSync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy2D,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy3D,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyAsync_V2
};

const std::vector<cnpapi_CallbackIdCNDRV> enabledCndrvCbidList = {
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel,
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernelEx,
  CNPAPI_CNDRV_TRACE_CBID_cnTaskTopoEntityInvoke,
  CNPAPI_CNDRV_TRACE_CBID_cnPlaceNotifier,
  CNPAPI_CNDRV_TRACE_CBID_cnPlaceNotifierWithFlags,
  CNPAPI_CNDRV_TRACE_CBID_cnQueueWaitNotifier,
  CNPAPI_CNDRV_TRACE_CBID_cnQueueWaitNotifierWithFlags,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoDAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoDAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoHAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoHAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyPeerAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoDAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy2D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy2DAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy3D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy3DAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyPeer,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoD,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoH,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD2D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD3D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD8Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD16Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD8,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD16,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32,
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeHostFunc,
  CNPAPI_CNDRV_TRACE_CBID_cnCnpapiInternalReserved1,
  CNPAPI_CNDRV_TRACE_CBID_cnQueueAtomicOperation,
  CNPAPI_CNDRV_TRACE_CBID_cnMalloc,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocSecurity,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocNode,
  CNPAPI_CNDRV_TRACE_CBID_cnZmalloc,
  CNPAPI_CNDRV_TRACE_CBID_cnZmallocNode,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocConstant,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocNodeConstant,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocFrameBuffer,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocPeerAble,
  CNPAPI_CNDRV_TRACE_CBID_cnFree
};

static std::vector<cnpapi_CallbackIdCNNL> getEnabledCnnlCbidList() {
  std::vector<cnpapi_CallbackIdCNNL> enabled_list;
  // Don't record cnnl apis whose name contain substrings in disable_match_ptr.
  std::vector<std::string> disable_match_ptr = {"WorkspaceSize",
                                                "Descriptor",
                                                "DescCreate",
                                                "DescDestroy",
                                                "DescAttr",
                                                "AlgoCreate",
                                                "AlgoDestroy",
                                                "cnnlSet",
                                                "cnnlGet",
                                                "cnnlCreate",
                                                "cnnlDestroy"};
  auto isMatched = [&](const std::string& name) {
    for (const auto& str : disable_match_ptr) {
      if (name.find(str) != std::string::npos) {
        return true;
      }
    }
    return false;
  };
  for (int i = 0; i < CNPAPI_CNNL_TRACE_CBID_SIZE; ++i) {
    cnpapi_CallbackIdCNNL cbid = static_cast<cnpapi_CallbackIdCNNL>(i);
    std::string name = runtimeCbidName(CNPAPI_ACTIVITY_TYPE_CNNL_API, cbid);
    if (!isMatched(name)) {
      enabled_list.push_back(cbid);
    }
  }
  return enabled_list;
}

// By default only some subscribed cnnl/cnrt/cndrv's apis are recorded.
// Set env KINETO_MLU_RECORD_ALL_APIS=1 can record all apis.
static bool RECORD_ALL_APIS = [] {
  bool result = false;
  const auto record_env = std::getenv("KINETO_MLU_RECORD_ALL_APIS");
  if (record_env) {
    std::string record_env_str = record_env;
    std::transform(record_env_str.begin(), record_env_str.end(), record_env_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (record_env_str == "true" || record_env_str == "1" || record_env_str == "on") {
      result = true;
    }
  }
  return result;
}();

}  // namespace

const std::vector<cnpapi_CallbackIdCNDRV>& CnpapiActivityApi::getCndrvLaunchCbidList() {
  return enabledCndrvCbidList;
}

void CnpapiActivityApi::enableCnpapiActivityTypes(bool enable) {
  const cnpapiActivityType enable_activity_types[] = {
    CNPAPI_ACTIVITY_TYPE_CNRT_API,
    CNPAPI_ACTIVITY_TYPE_CNDRV_API,
    CNPAPI_ACTIVITY_TYPE_CNNL_API,
    CNPAPI_ACTIVITY_TYPE_CNNL_EXTRA_API,
    CNPAPI_ACTIVITY_TYPE_CNCL_API
  };
  for (const auto type : enable_activity_types) {
    if (enable) {
      CNPAPI_CALL(cnpapiActivityEnable(type));
    } else {
      CNPAPI_CALL(cnpapiActivityDisable(type));
    }
  }
}

#endif

void CnpapiActivityApi::enableCnpapiActivities(
    const std::set<ActivityType>& selected_activities) {
  all_records_ = std::make_unique<CnpapiRecordSet>();
#ifdef HAS_CNPAPI
  static bool registered = false;
  if (!registered) {
    CNPAPI_CALL(
        cnpapiActivityRegisterCallbacks(bufferRequestedTrampoline, bufferCompletedTrampoline));
    registered = true;
  }

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::MLU_MEMCPY) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
    }
    if (activity == ActivityType::MLU_MEMSET) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    }
    if (activity == ActivityType::MLU_CONCURRENT_KERNEL) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_ATOMIC_OPERATION));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION));
      externalCorrelationEnabled_ = true;
    }
    if (activity == ActivityType::MLU_RUNTIME) {
      if (RECORD_ALL_APIS) {
        enableCnpapiActivityTypes(true);
      } else {
        static const std::vector<cnpapi_CallbackIdCNNL> enabledCnnlCbidList = getEnabledCnnlCbidList();
        CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CALLBACK_API));
        CNPAPI_CALL(cnpapiSubscribe(&subscriber_, (cnpapi_CallbackFunc)emptyCallback, nullptr));
        CNPAPI_CALL(cnpapiEnableDomain(1, subscriber_, CNPAPI_CB_DOMAIN_CNNL_EXTRA_API));
        CNPAPI_CALL(cnpapiEnableDomain(1, subscriber_, CNPAPI_CB_DOMAIN_CNCL_API));
        for (const auto& cbid : enabledCnrtCbidList) {
          CNPAPI_CALL(cnpapiEnableCallback(1, subscriber_, CNPAPI_CB_DOMAIN_CNRT_API, cbid));
        }
        for (const auto& cbid : enabledCndrvCbidList) {
          CNPAPI_CALL(cnpapiEnableCallback(1, subscriber_, CNPAPI_CB_DOMAIN_CNDRV_API, cbid));
        }
        for (const auto& cbid : enabledCnnlCbidList) {
          CNPAPI_CALL(cnpapiEnableCallback(1, subscriber_, CNPAPI_CB_DOMAIN_CNNL_API, cbid));
        }
      }
    }
    if (activity == ActivityType::OVERHEAD) {
      CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_OVERHEAD));
    }
  }
#endif

  // Explicitly enabled, so reset this flag if set
  stopCollection = false;
}

void CnpapiActivityApi::disableCnpapiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_CNPAPI
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::MLU_MEMCPY) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP));
    }
    if (activity == ActivityType::MLU_MEMSET) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    }
    if (activity == ActivityType::MLU_CONCURRENT_KERNEL) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_KERNEL));
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_ATOMIC_OPERATION));
    }
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION));
    }
    if (activity == ActivityType::MLU_RUNTIME) {
      if (RECORD_ALL_APIS) {
        enableCnpapiActivityTypes(false);
      } else {
        CNPAPI_CALL(cnpapiUnsubscribe(subscriber_));
        CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CALLBACK_API));
      }
    }
    if (activity == ActivityType::OVERHEAD) {
      CNPAPI_CALL(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_OVERHEAD));
    }
  }
  externalCorrelationEnabled_ = false;
#endif
}

} // namespace KINETO_NAMESPACE
