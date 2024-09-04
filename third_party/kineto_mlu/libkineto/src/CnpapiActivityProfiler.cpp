/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CnpapiActivityProfiler.h"

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <chrono>

#include "CnpapiActivity.h"
#include "Demangle.h"
#include "Logger.h"

using namespace std::chrono;

namespace libkineto_mlu {

/* ----------------------------------------
 * Implement CnpapiActivityProfilerSession
 * ----------------------------------------
 */

CnpapiActivityProfilerSession::CnpapiActivityProfilerSession(
    const Config& config, CnpapiActivityApi& cnpapi)
     : cnpapi_(cnpapi),
       flushOverhead_{0, 0},
       setupOverhead_{0, 0},
       selectedActivityTypes_(config.selectedActivityTypes()) {
  resetTraceData();
  traceBuffer_.span = {0, 0, "MLU Kernels", ""};
  cnpapiBuffers_ = std::make_unique<CnpapiRecordSet>();
  cnpapi_.setMaxBufferSize(config.activitiesMaxGpuBufferSize());

  time_point<system_clock> timestamp;
  if (VLOG_IS_ON(1)) {
    timestamp = system_clock::now();
  }
  cnpapi_.enableCnpapiActivities(config.selectedActivityTypes());
  if (VLOG_IS_ON(1)) {
    auto t2 = system_clock::now();
    addOverheadSample(
        setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
  }
}

void CnpapiActivityProfilerSession::stop() {
  time_point<system_clock> timestamp;
  if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
  }
  cnpapi_.disableCnpapiActivities(selectedActivityTypes_);
  if (VLOG_IS_ON(1)) {
    auto t2 = system_clock::now();
    addOverheadSample(
        setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
  }
}

void CnpapiActivityProfilerSession::processTrace(ActivityLogger& logger,
        getLinkedActivityCallback getLinkedActivity,
        int64_t startTime, int64_t endTime) {
  captureWindowStartTime_ = startTime;
  captureWindowEndTime_ = endTime;
  getLinkedActivity_ = getLinkedActivity;
  LOG(INFO) << "Retrieving MLU activity buffers";
  cnpapiBuffers_ = cnpapi_.activityBuffers();
  if (VLOG_IS_ON(1)) {
      addOverheadSample(flushOverhead_, cnpapi_.flushOverhead);
  }
  if (cnpapiBuffers_) {
    // Handle external_correlation first
    for (const auto& record : cnpapiBuffers_->external_correlation_records) {
      handleCorrelationActivity(record);
    }
    for (const auto& record : cnpapiBuffers_->runtime_records) {
      handleRuntimeActivity(record, &logger);
    }
    for (const auto& record : cnpapiBuffers_->kernel_records) {
      handleMluActivity(record, &logger);
    }
    for (const auto& record : cnpapiBuffers_->memcpy_records) {
      handleMluActivity(record, &logger);
    }
    for (const auto& record : cnpapiBuffers_->memcpy_p2p_records) {
      handleMluActivity(record, &logger);
    }
    for (const auto& record : cnpapiBuffers_->memset_records) {
      handleMluActivity(record, &logger);
    }
    for (const auto& record : cnpapiBuffers_->overhead_records) {
      handleOverheadActivity(record, &logger);
    }
    for (const auto& record : cnpapiBuffers_->atomic_op_records) {
      handleMluActivity(record, &logger);
    }
    // resourceOverheadCount_ is set while processing MLU activities
    if (resourceOverheadCount_ > 0) {
      LOG(INFO) << "Allocated " << resourceOverheadCount_ << " extra CNPAPI buffers.";
    }
    LOGGER_OBSERVER_ADD_METADATA("ResourceOverhead", std::to_string(resourceOverheadCount_));
  }
  mluUserEventMap_.logEvents(&logger);
}

std::vector<std::string> CnpapiActivityProfilerSession::errors() {
  return {};
}

std::unique_ptr<DeviceInfo> CnpapiActivityProfilerSession::getDeviceInfo() {
  int32_t pid = processId();
  std::string process_name = processName(pid);
  int32_t device_id = -1;
  if (resourceInfo_.size() > 0) {
    device_id = resourceInfo_.begin()->first.first;
  }

  return std::make_unique<DeviceInfo>(device_id, process_name, "MLU");
}

std::vector<ResourceInfo> CnpapiActivityProfilerSession::getResourceInfos() {
  std::vector<ResourceInfo> res_info;
  for (const auto& info : resourceInfo_) {
    res_info.push_back(info.second);
  }
  return res_info;
}

void CnpapiActivityProfilerSession::pushCorrelationId(uint64_t id) {
  CnpapiActivityApi::pushCorrelationID(id,
    CnpapiActivityApi::CorrelationFlowType::Default);
}

void CnpapiActivityProfilerSession::popCorrelationId() {
  CnpapiActivityApi::popCorrelationID(
    CnpapiActivityApi::CorrelationFlowType::Default);
}

void CnpapiActivityProfilerSession::pushUserCorrelationId(uint64_t id) {
  CnpapiActivityApi::pushCorrelationID(id,
    CnpapiActivityApi::CorrelationFlowType::User);
}

void CnpapiActivityProfilerSession::popUserCorrelationId() {
  CnpapiActivityApi::popCorrelationID(
    CnpapiActivityApi::CorrelationFlowType::User);
}

static GenericTraceActivity createUserMluSpan(
    const ITraceActivity& cpuTraceActivity,
    const ITraceActivity& mluTraceActivity) {
  GenericTraceActivity res(
      *cpuTraceActivity.traceSpan(),
      ActivityType::GPU_USER_ANNOTATION,
      cpuTraceActivity.name());
  res.startTime = mluTraceActivity.timestamp();
  res.device = mluTraceActivity.deviceId();
  res.resource = mluTraceActivity.resourceId();
  res.endTime =
      mluTraceActivity.timestamp() + mluTraceActivity.duration();
  res.id = cpuTraceActivity.correlationId();
  return res;
}

void CnpapiActivityProfilerSession::MluUserEventMap::insertOrExtendEvent(
    const ITraceActivity& userActivity,
    const ITraceActivity& mluActivity) {
  StreamKey key(mluActivity.deviceId(), mluActivity.resourceId());
  CorrelationSpanMap& correlationSpanMap = streamSpanMap_[key];
  auto it = correlationSpanMap.find(userActivity.correlationId());
  if (it == correlationSpanMap.end()) {
    auto it_success = correlationSpanMap.insert({
        userActivity.correlationId(), createUserMluSpan(userActivity, mluActivity)
    });
    it = it_success.first;
  }
  GenericTraceActivity& span = it->second;
  if (mluActivity.timestamp() < span.startTime || span.startTime == 0) {
    span.startTime = mluActivity.timestamp();
  }
  int64_t mlu_activity_end = mluActivity.timestamp() + mluActivity.duration();
  if (mlu_activity_end > span.endTime) {
    span.endTime = mlu_activity_end;
  }
}

void CnpapiActivityProfilerSession::MluUserEventMap::logEvents(ActivityLogger *logger) {
  for (auto const& streamMapPair : streamSpanMap_) {
    for (auto const& correlationSpanPair : streamMapPair.second) {
      correlationSpanPair.second.log(*logger);
    }
  }
}

inline bool CnpapiActivityProfilerSession::outOfRange(const ITraceActivity* act) {
  bool out_of_range = act->timestamp() < captureWindowStartTime_ ||
      (act->timestamp() + act->duration()) > captureWindowEndTime_;
  if (out_of_range) {
    LOG(WARNING) << "TraceActivity outside of profiling window: " << act->name()
        << " (" << act->timestamp() << " < " << captureWindowStartTime_ << " or "
        << (act->timestamp() + act->duration()) << " > " << captureWindowEndTime_;
  }
  return out_of_range;
}

inline void CnpapiActivityProfilerSession::updateMluNetSpan(
    const GenericTraceActivity* mlu_act) {
  if (!mlu_act->linkedActivity()) {
    LOG(WARNING) << "Missing linked activity";
    return;
  }
  TraceSpan& mlu_span = traceBuffer_.span;
  if (mlu_act->timestamp() < mlu_span.startTime || mlu_span.startTime == 0) {
    mlu_span.startTime = mlu_act->timestamp();
  }
  if ((mlu_act->timestamp() + mlu_act->duration()) > mlu_span.endTime) {
    mlu_span.endTime = mlu_act->timestamp() + mlu_act->duration();
  }
}

// I've observed occasional broken timestamps attached to MLU events...
void CnpapiActivityProfilerSession::checkTimestampOrder(const ITraceActivity* act1) {
  // Correlated MLU runtime activity cannot
  // have timestamp greater than the MLU activity's
  const auto& it = correlatedMluActivities_.find(act1->correlationId());
  if (it == correlatedMluActivities_.end()) {
    correlatedMluActivities_.insert({act1->correlationId(), act1});
    return;
  }

  // Activities may be appear in the buffers out of order.
  // If we have a runtime activity in the map, it should mean that we
  // have a MLU activity passed in, and vice versa.
  const ITraceActivity* act2 = it->second;
  if (act2->type() == ActivityType::PRIVATEUSE1_RUNTIME) {
    // Buffer is out-of-order.
    // Swap so that runtime activity is first for the comparison below.
    std::swap(act1, act2);
  }
  if (act1->timestamp() > act2->timestamp()) {
    LOG(WARNING) << "MLU op timestamp (" << act2->timestamp()
                 << ") < runtime timestamp (" << act1->timestamp() << ") by "
                 << act1->timestamp() - act2->timestamp() << "us";
    LOG(WARNING) << "Name: " << act2->name()
                 << " Device: " << act2->deviceId()
                 << " Stream: " << act2->resourceId();
  }
}

inline void CnpapiActivityProfilerSession::handleCorrelationActivity(
    const ExternalCorrelationRecord& correlation) {
  if (correlation.external_type == CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0) {
    cpuCorrelationMap_[correlation.correlation_id] = correlation.external_id;
  } else if (correlation.external_type == CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1){
    userCorrelationMap_[correlation.correlation_id] = correlation.external_id;
  } else {
    LOG(WARNING) << "Invalid cnpapiActivityExternalCorrelation sent to handleCnpapiActivity";
  }
}

const ITraceActivity* CnpapiActivityProfilerSession::linkedActivity(
    int32_t correlation_id,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlation_id);
  if (it != correlationMap.end()) {
    return getLinkedActivity_(it->second);
  }
  return nullptr;
}

void CnpapiActivityProfilerSession::handleRuntimeActivity(
    const RuntimeRecord& activity,
    ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(
      activity.correlation_id, cpuCorrelationMap_);

  LOG(INFO) << activity.correlation_id
            << ": CNPAPI_ACTIVITY_KIND_RUNTIME, cbid=" << activity.cbid
            << " tid=" << activity.thread_id;
  int32_t tid = activity.thread_id;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const auto* runtime_activity =
      addActivityWrapper(createRuntimeActivity(&activity, linked, tid));
  checkTimestampOrder(runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  logger->handleActivity(*runtime_activity);
}

void CnpapiActivityProfilerSession::handleOverheadActivity(
    const OverheadRecord& activity,
    ActivityLogger* logger) {
  LOG(INFO) << ": CNPAPI_ACTIVITY_KIND_OVERHEAD" << " overheadKind=" << activity.overhead_type;
  // Monitor memory overhead
  if (activity.overhead_type == CNPAPI_ACTIVITY_OVERHEAD_CNPAPI_RESOURCE) {
    resourceOverheadCount_++;
  }
  const auto* overhead_activity =
      addActivityWrapper(createOverheadActivity(&activity, nullptr, 0));
  if (outOfRange(overhead_activity)) {
    return;
  }
  logger->handleActivity(*overhead_activity);
}

inline void CnpapiActivityProfilerSession::handleMluActivity(
    const GenericTraceActivity* act,
    ActivityLogger* logger) {
  if (outOfRange(act)) {
    return;
  }
  traceBuffer_.span.opCount += 1;
  checkTimestampOrder(act);
  LOG(INFO) << act->correlationId() << ": "
            << act->name();
  recordStream(act->deviceId(), act->resourceId(), "");
  if (selectedActivityTypes_.count(ActivityType::GPU_USER_ANNOTATION)) {
    const auto& it = userCorrelationMap_.find(act->correlationId());
    if (it != userCorrelationMap_.end()) {
      auto linked_act = getLinkedActivity_(it->second);
      if (linked_act) {
        recordStream(act->deviceId(), act->resourceId(), "context");
        mluUserEventMap_.insertOrExtendEvent(*linked_act, *act);
      }
    }
  }
  updateMluNetSpan(act);
  logger->handleActivity(*act);
}

template <class T>
inline void CnpapiActivityProfilerSession::handleMluActivity(
    const T& act, ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(
      act.correlation_id, cpuCorrelationMap_);

  const auto* mlu_activity = addActivityWrapper(createMluActivity(&act, linked));
  handleMluActivity(mlu_activity, logger);
}

void CnpapiActivityProfilerSession::resetTraceData() {
  cnpapi_.clearActivities();
  cpuCorrelationMap_.clear();
  correlatedMluActivities_.clear();
  mluUserEventMap_.clear();
  cnpapiBuffers_ = nullptr;
  resourceOverheadCount_ = 0;
  traceBuffer_.activities.clear();
  traceBuffer_.span = {0, 0, "none", ""};
}


/* ----------------------------------------
 * Implement CnpapiActivityProfiler
 * ----------------------------------------
 */

namespace {

bool hasTestEnvVar() {
  return getenv("GTEST_OUTPUT") != nullptr || getenv("FB_TEST") != nullptr
     || getenv("PYTORCH_TEST") != nullptr || getenv("TEST_PILOT") != nullptr;
}

void suppressLibkinetoMLULogMessages() {
  // Only suppress messages if explicit override wasn't provided
  const char* logLevelEnv = getenv("KINETO_MLU_LOG_LEVEL");
  // For unit tests, don't suppress log verbosity.
  if (!hasTestEnvVar() && (!logLevelEnv || !*logLevelEnv)) {
    SET_LOG_SEVERITY_LEVEL(ERROR);
  }
  if (logLevelEnv) {
    // atoi returns 0 on error, so that's what we want - default to VERBOSE
    static_assert (static_cast<int>(VERBOSE) == 0, "");
    SET_LOG_SEVERITY_LEVEL(atoi(logLevelEnv));
  }
}

}  // namespace

CnpapiActivityProfiler::CnpapiActivityProfiler() {
  suppressLibkinetoMLULogMessages();
}

const std::string kProfilerName{"CnpapiActivityProfiler"};

const std::string& CnpapiActivityProfiler::name() const {
  return kProfilerName;
}

const std::set<ActivityType>& CnpapiActivityProfiler::availableActivities()
    const {
  static const std::set<ActivityType> supportted_activities =
                                {ActivityType::GPU_USER_ANNOTATION,
                                 ActivityType::GPU_MEMCPY,
                                 ActivityType::GPU_MEMSET,
                                 ActivityType::CONCURRENT_KERNEL,
                                 ActivityType::EXTERNAL_CORRELATION,
                                 ActivityType::PRIVATEUSE1_RUNTIME,
                                 ActivityType::OVERHEAD};
  return supportted_activities;
}

std::unique_ptr<IActivityProfilerSession> CnpapiActivityProfiler::configure(
    const std::set<ActivityType>& /*activity_types*/,
    const Config& config) {
  LOG(INFO) << "Enabling MLU tracing";
  return std::make_unique<CnpapiActivityProfilerSession>(config,
            CnpapiActivityApi::singleton());
  return nullptr;
}

std::unique_ptr<IActivityProfilerSession>
CnpapiActivityProfiler::configure(
    int64_t /*ts_ms*/,
    int64_t /*duration_ms*/,
    const std::set<ActivityType>& activity_types,
    const Config& config) {
  return configure(activity_types, config);
};

/* ----------------------------------------
 * Register the CnpapiActivityProfiler creator
 * ----------------------------------------
 */
CnpapiActivityProfilerInit::CnpapiActivityProfilerInit() {
  libkineto::api().registerProfilerFactory([&]() {
    return std::make_unique<CnpapiActivityProfiler>();
  });
}

static std::unique_ptr<CnpapiActivityProfilerInit>
      initInstance = std::make_unique<CnpapiActivityProfilerInit>();

}  // namespace libkineto_mlu
