// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>

#ifdef HAS_CNPAPI
#include <cnpapi.h>
#endif

#include "cnpapi_call.h"
#include "ActivityType.h"
#include "CnpapiActivityBuffer.h"
#include "CnpapiRecord.h"


namespace KINETO_NAMESPACE {

using namespace libkineto;

#ifndef HAS_CNPAPI
using cnpapiActivity = void;
#endif

class CnpapiActivityApi {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  CnpapiActivityApi() {
#ifdef HAS_CNPAPI
    CNPAPI_CALL(cnpapiInit());
#endif
  }
  CnpapiActivityApi(const CnpapiActivityApi&) = delete;
  CnpapiActivityApi& operator=(const CnpapiActivityApi&) = delete;

  virtual ~CnpapiActivityApi() {
#ifdef HAS_CNPAPI
    CNPAPI_CALL(cnpapiRelease());
#endif
  }

  static CnpapiActivityApi& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableCnpapiActivities(
    const std::set<ActivityType>& selected_activities);
  void disableCnpapiActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();

  std::unique_ptr<CnpapiRecordSet> activityBuffers();

  void setMaxBufferSize(int size);

#ifdef HAS_CNPAPI
  const std::vector<cnpapi_CallbackIdCNDRV>& getCndrvLaunchCbidList();
#endif

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
#ifdef HAS_CNPAPI
  int processActivitiesForBuffer(
      uint64_t* buf,
      size_t validSize);
  static void
  bufferRequestedTrampoline(uint64_t** buffer, size_t* size, size_t* maxNumRecords);
  static void bufferCompletedTrampoline(
      uint64_t* buffer,
      size_t size,
      size_t validSize);
  cnpapi_SubscriberHandle subscriber_;
  static void emptyCallback(void *userdata, cnpapi_CallbackDomain domain,
                     cnpapi_CallbackId cbid, const cnpapi_CallbackData *cbdata) {}
  void enableCnpapiActivityTypes(bool enable);
#endif // HAS_CNPAPI

  int maxMluBufferCount_{0};
  CnpapiActivityBufferMap allocatedMluTraceBuffers_;
  std::mutex mutex_;
  bool externalCorrelationEnabled_{false};
  std::unique_ptr<CnpapiRecordSet> all_records_;

 protected:
#ifdef HAS_CNPAPI
  void bufferRequested(uint64_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(
      uint64_t* buffer,
      size_t size,
      size_t validSize);
  void processRecords(cnpapiActivity* raw_record);
#endif
};

} // namespace KINETO_NAMESPACE
