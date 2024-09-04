/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <cnpapi.h>

#include "cnpapi_call.h"
#include "ActivityType.h"
#include "CnpapiActivityBuffer.h"
#include "CnpapiRecord.h"


namespace libkineto_mlu {

using libkineto::ActivityType;

class CnpapiActivityApi {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  CnpapiActivityApi() {
    CNPAPI_CALL(cnpapiInit());
  }
  CnpapiActivityApi(const CnpapiActivityApi&) = delete;
  CnpapiActivityApi& operator=(const CnpapiActivityApi&) = delete;

  virtual ~CnpapiActivityApi() {
    CNPAPI_CALL(cnpapiRelease());
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

  void enableCnpapiActivityTypes(bool enable);
  
  void setMaxBufferSize(int size);

  const std::vector<cnpapi_CallbackIdCNDRV>& getCndrvLaunchCbidList();

  std::atomic_bool stopCollection{false};
  int64_t flushOverhead{0};

 private:
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

  int maxMluBufferCount_{0};
  CnpapiActivityBufferMap allocatedMluTraceBuffers_;
  std::unique_ptr<CnpapiActivityBufferMap> readyMluTraceBuffers_;
  std::mutex mutex_;
  bool externalCorrelationEnabled_{false};
  std::unique_ptr<CnpapiRecordSet> all_records_;

 protected:
  void bufferRequested(uint64_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(
      uint64_t* buffer,
      size_t size,
      size_t validSize);
  void processRecords(cnpapiActivity* raw_record);
};

} // namespace libkineto_mlu
