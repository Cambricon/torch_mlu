// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CnpapiActivityApi.h"

#include <assert.h>
#include <chrono>

#include "ApiListCommon.h"
#include "CnpapiCallbackManager.h"
#include "CnpapiResourceApi.h"
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
      CnpapiResourceApi::singleton().pushCorrelationID(id);
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
      CnpapiResourceApi::singleton().popCorrelationID();
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

void CnpapiActivityApi::subscribedCallback(
    void* userdata,
    cnpapi_CallbackDomain domain,
    cnpapi_CallbackId cbid,
    const void* cbdata) {
  for (const auto& callback : singleton().callback_func_list_) {
    (*callback)(userdata, domain, cbid, cbdata);
  }
}

void CnpapiActivityApi::updateCallbackFunction() {
  callback_func_list_.clear();
  for (const auto& callback :
      kineto_mlu::CnpapiCallbackManager::getInstance()
          .getCallbackFunctions()) {
    callback_func_list_.push_back(callback.get());
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
  updateCallbackFunction();
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
        ApiList& apiList = ApiList::getInstance();
        const std::vector<cnpapi_CallbackIdCNNL> enabledCnnlCbidList = apiList.getCnnlCbidList();
        const std::vector<cnpapi_CallbackIdCNRT> enabledCnrtCbidList = apiList.getCnrtCbidList();
        const std::vector<cnpapi_CallbackIdCNDRV> enabledCndrvCbidList = apiList.getCndrvCbidList();

        CNPAPI_CALL(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CALLBACK_API));
        CNPAPI_CALL(cnpapiSubscribe(&subscriber_, (cnpapi_CallbackFunc)subscribedCallback, nullptr));
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
        for (const auto& cbid_pair :
            kineto_mlu::CnpapiCallbackManager::getInstance().getEnabledCbids()) {
          // first: domain
          // second: [cbid1, cbid2, ...]
          for (const auto& cbid : cbid_pair.second) {
            CNPAPI_CALL(cnpapiEnableCallback(1, subscriber_, cbid_pair.first, cbid));
          }
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
