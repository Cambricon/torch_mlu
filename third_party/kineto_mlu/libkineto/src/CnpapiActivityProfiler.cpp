// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CnpapiActivityProfiler.h"

#include <fmt/format.h>
#include <time.h>
#include <atomic>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>
#include <limits>

#ifdef HAS_CNPAPI
#include <cnpapi.h>
// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "mlu_call.h"
#include "cnpapi_call.h"
#endif

#include "Config.h"
#include "time_since_epoch.h"
#ifdef HAS_CNPAPI
#include "CnpapiActivity.h"
#include "CnpapiActivity.tpp"
#include "CnpapiActivityApi.h"
#include "CnpapiResourceApi.h"
#endif // HAS_CNPAPI
#include "output_base.h"

#include "Logger.h"
#include "ThreadUtil.h"

using namespace std::chrono;
using namespace libkineto;
using std::string;

namespace KINETO_NAMESPACE {

ConfigDerivedState::ConfigDerivedState(const Config& config) {
  profileActivityTypes_ = config.selectedActivityTypes();
  profileStartTime_ = config.requestTimestamp();
  profileDuration_ = config.activitiesDuration();
  profileWarmupDuration_ = config.activitiesWarmupDuration();
  profilingByIter_ = config.hasProfileStartIteration();
  if (profilingByIter_) {
    profileStartIter_ = config.profileStartIteration();
    profileEndIter_ = profileStartIter_ + config.activitiesRunIterations();
  } else {
    profileEndIter_ = (std::numeric_limits<decltype(profileEndIter_)>::max)();
    profileEndTime_ = profileStartTime_ + config.activitiesDuration();
  }
}

bool ConfigDerivedState::canStart(
    const std::chrono::time_point<std::chrono::system_clock>& now) const {
  if (profilingByIter_) {
    return true;
  }
  if (profileStartTime_ < now) {
    LOG(ERROR) << "Not starting tracing - start timestamp is in the past. Time difference (ms): "
                << duration_cast<milliseconds>(now - profileStartTime_).count();
    return false;
  } else if ((profileStartTime_ - now) < profileWarmupDuration_) {
    LOG(ERROR) << "Not starting tracing - insufficient time for warmup. Time to warmup (ms): "
                << duration_cast<milliseconds>(profileStartTime_ - now).count();
    return false;
  }
  return true;
}

bool ConfigDerivedState::isWarmupDone(
      const time_point<system_clock>& now,
      int64_t currentIter) const {
  bool isTimestampBased = !profilingByIter_ && currentIter < 0;
  if (isTimestampBased) {
    // qualify that this check is not being called from application step() API
    // this avoids races between the step() API and periodically invoked
    // profiler run loop step() method
    return now >= profileStartTime_;
  }
  bool isIterationBased = profilingByIter_ && currentIter >= 0;
  if (isIterationBased) {
    return currentIter >= profileStartIter_;
  }
  return false;
}

bool ConfigDerivedState::isCollectionDone(
      const time_point<system_clock>& now,
      int64_t currentIter) const {
  bool isTimestampBased = !profilingByIter_ && currentIter < 0;
  if (isTimestampBased) {
    // qualify that this check is not being called from application step() API
    return now >= profileEndTime_;
  }
  bool isIterationBased = profilingByIter_ && currentIter >= 0;
  if (isIterationBased) {
    return currentIter >= profileEndIter_;
  }
  return false;
}

void CnpapiActivityProfiler::transferCpuTrace(
    std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace) {
  std::lock_guard<std::mutex> guard(mutex_);
  const string& trace_name = cpuTrace->span.name;
  if (currentRunloopState_ != RunloopState::CollectTrace &&
      currentRunloopState_ != RunloopState::ProcessTrace) {
    VLOG(0) << "Trace collection not in progress - discarding span "
            << trace_name;
    return;
  }

  cpuTrace->span.iteration = iterationCountMap_[trace_name]++;

  VLOG(0) << "Received iteration " << cpuTrace->span.iteration << " of span "
          << trace_name << " (" << cpuTrace->activities.size() << " activities / "
          << cpuTrace->mluOpCount << " mlu activities)";
  traceBuffers_->cpu.push_back(std::move(cpuTrace));
}

CnpapiActivityProfiler::CnpapiActivityProfiler(CnpapiActivityApi& cnpapi, bool cpuOnly)
    : cnpapi_(cnpapi),
      flushOverhead_{0, 0},
      setupOverhead_{0, 0},
      cpuOnly_{cpuOnly},
      currentRunloopState_{RunloopState::WaitForRequest} {

#ifdef HAS_CNPAPI
    // determine MLU availability on the system
    cnrtRet_t error;
    unsigned int deviceCount;
    error = cnrtGetDeviceCount(&deviceCount);
    bool mluAvailable = (error == cnrtSuccess && deviceCount > 0);

    if (mluAvailable) {
      logMluVersions();
    }

    static std::unique_ptr<CnpapiResourceCallbackRegistration>
      resourceRegisterInstance = std::make_unique<CnpapiResourceCallbackRegistration>();

    static std::unique_ptr<CnpapiPmuCallbackRegistration>
      pmuRegisterInstance = std::make_unique<CnpapiPmuCallbackRegistration>();
#endif
  }

#ifdef HAS_CNPAPI
void CnpapiActivityProfiler::logMluVersions() {
  // check mlu versions
  int cnpapiMajor, cnpapiMinor, cnpapiPatch;
  int cnrtMajor, cnrtMinor, cnrtPatch;
  int cndrvMajor, cndrvMinor, cndrvPatch;

  CNPAPI_CALL(cnpapiGetLibVersion(&cnpapiMajor, &cnpapiMinor, &cnpapiPatch));
  CNRT_CALL(cnrtGetLibVersion(&cnrtMajor, &cnrtMinor, &cnrtPatch));
  CNRT_CALL(cnrtDriverGetVersion(&cndrvMajor, &cndrvMinor, &cndrvPatch));

  auto fmtVer = [=](int major, int minor, int patch) {
    return fmt::format("{}", fmt::join(
      std::vector<std::string>({std::to_string(major),
                                std::to_string(minor),
                                std::to_string(patch)}), "."));
  };
  LOG(INFO) << "MLU versions. CNPapi: " << fmtVer(cnpapiMajor, cnpapiMinor, cnpapiPatch)
            << "; Runtime: " << fmtVer(cnrtMajor, cnrtMinor, cnrtPatch)
            << "; Driver: " << fmtVer(cndrvMajor, cndrvMinor, cndrvPatch);
}
#endif

void CnpapiActivityProfiler::processTraceInternal(ActivityLogger& logger) {
  LOG(INFO) << "Processing " << traceBuffers_->cpu.size() << " CPU buffers";
  VLOG(0) << "Profile time range: " << captureWindowStartTime_ << " - "
          << captureWindowEndTime_;
  logger.handleTraceStart(metadata_);
  for (auto& cpu_trace : traceBuffers_->cpu) {
    string trace_name = cpu_trace->span.name;
    VLOG(0) << "Processing CPU buffer for " << trace_name << " ("
            << cpu_trace->span.iteration << ") - "
            << cpu_trace->activities.size() << " records";
    VLOG(0) << "Span time range: " << cpu_trace->span.startTime << " - "
            << cpu_trace->span.endTime;
    processCpuTrace(*cpu_trace, logger);
    LOGGER_OBSERVER_ADD_EVENT_COUNT(cpu_trace->activities.size());
  }

#ifdef HAS_CNPAPI
  if (!cpuOnly_) {
    VLOG(0) << "Retrieving MLU activity buffers";
    traceBuffers_->mlu = cnpapi_.activityBuffers();
    pmuDataMap_ = CnpapiPmuApi::singleton().getPmuData();
    if (VLOG_IS_ON(1)) {
      addOverheadSample(flushOverhead_, cnpapi_.flushOverhead);
    }
    if (traceBuffers_->mlu) {
      CnpapiResourceApi::singleton().processTaskTopoData(activityMap_);
      // Handle external_correlation first
      for (const auto& record : traceBuffers_->mlu->external_correlation_records) {
        handleCorrelationActivity(record);
      }
      for (const auto& record : traceBuffers_->mlu->runtime_records) {
        handleRuntimeActivity(record, &logger);
      }
      for (const auto& record : traceBuffers_->mlu->kernel_records) {
        std::vector<CnpapiPmuData>* pmu_data = nullptr;
        if (pmuDataMap_) {
          auto it = pmuDataMap_->find(record.correlation_id);
          if (it != pmuDataMap_->end()) {
            pmu_data = &it->second;
          }
        }
        handleMluActivity(record, pmu_data, &logger);
      }
      for (const auto& record : traceBuffers_->mlu->memcpy_records) {
        handleMluActivity(record, nullptr, &logger);
      }
      for (const auto& record : traceBuffers_->mlu->memcpy_p2p_records) {
        handleMluActivity(record, nullptr, &logger);
      }
      for (const auto& record : traceBuffers_->mlu->memset_records) {
        handleMluActivity(record, nullptr, &logger);
      }
      for (const auto& record : traceBuffers_->mlu->overhead_records) {
        handleOverheadActivity(record, &logger);
      }
      for (const auto& record : traceBuffers_->mlu->atomic_op_records) {
        handleMluActivity(record, nullptr, &logger);
      }
      // resourceOverheadCount_ is set while processing MLU activities
      if (resourceOverheadCount_ > 0) {
        LOG(INFO) << "Allocated " << resourceOverheadCount_ << " extra CNPAPI buffers.";
      }
      LOGGER_OBSERVER_ADD_METADATA("ResourceOverhead", std::to_string(resourceOverheadCount_));
    }
  }
#endif // HAS_CNPAPI

  for (const auto& session : sessions_){
    LOG(INFO) << "Processing child profiler trace";
    session->processTrace(logger);
  }

  finalizeTrace(*config_, logger);
}

CnpapiActivityProfiler::CpuMluSpanPair& CnpapiActivityProfiler::recordTraceSpan(
    TraceSpan& span, int mluOpCount) {
  TraceSpan mlu_span(mluOpCount, span.iteration, span.name, "MLU: ");
  auto& iterations = traceSpans_[span.name];
  iterations.push_back({span, mlu_span});
  return iterations.back();
}

void CnpapiActivityProfiler::processCpuTrace(
    libkineto::CpuTraceBuffer& cpuTrace,
    ActivityLogger& logger) {
  if (cpuTrace.activities.size() == 0) {
    LOG(WARNING) << "CPU trace is empty!";
    return;
  }

  CpuMluSpanPair& span_pair = recordTraceSpan(cpuTrace.span, cpuTrace.mluOpCount);
  TraceSpan& cpu_span = span_pair.first;
  for (auto const& act : cpuTrace.activities) {
    VLOG(2) << act->correlationId() << ": OP " << act->activityName;
    if (derivedConfig_->profileActivityTypes().count(act->type())) {
      act->log(logger);
    }
    clientActivityTraceMap_[act->correlationId()] = &span_pair;
    activityMap_[act->correlationId()] = act.get();

    recordThreadInfo(act->resourceId(), act->getThreadId(), act->deviceId());
  }
  logger.handleTraceSpan(cpu_span);
}

#ifdef HAS_CNPAPI
inline void CnpapiActivityProfiler::handleCorrelationActivity(
    const ExternalCorrelationRecord& correlation) {
  if (correlation.external_type == CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0) {
    cpuCorrelationMap_[correlation.correlation_id] = correlation.external_id;
  } else if (correlation.external_type == CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1){
    userCorrelationMap_[correlation.correlation_id] = correlation.external_id;
  } else {
    LOG(WARNING) << "Invalid cnpapiActivityExternalCorrelation sent to handleCnpapiActivity";
  }
}
#endif // HAS_CNPAPI

static GenericTraceActivity createUserMluSpan(
    const libkineto::ITraceActivity& cpuTraceActivity,
    const libkineto::ITraceActivity& mluTraceActivity) {
  GenericTraceActivity res(
      *cpuTraceActivity.traceSpan(),
      ActivityType::MLU_USER_ANNOTATION,
      cpuTraceActivity.name());
  res.startTime = mluTraceActivity.timestamp();
  res.device = mluTraceActivity.deviceId();
  res.resource = mluTraceActivity.resourceId();
  res.endTime =
      mluTraceActivity.timestamp() + mluTraceActivity.duration();
  res.id = cpuTraceActivity.correlationId();
  return res;
}

void CnpapiActivityProfiler::MluUserEventMap::insertOrExtendEvent(
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

const CnpapiActivityProfiler::CpuMluSpanPair& CnpapiActivityProfiler::defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  static CpuMluSpanPair span_pair(span, span);
  return span_pair;
}

void CnpapiActivityProfiler::MluUserEventMap::logEvents(ActivityLogger *logger) {
  for (auto const& streamMapPair : streamSpanMap_) {
    for (auto const& correlationSpanPair : streamMapPair.second) {
      correlationSpanPair.second.log(*logger);
    }
  }
}

#ifdef HAS_CNPAPI
inline bool CnpapiActivityProfiler::outOfRange(const ITraceActivity& act) {
  bool out_of_range = act.timestamp() < captureWindowStartTime_ ||
      (act.timestamp() + act.duration()) > captureWindowEndTime_;
  if (out_of_range) {
    VLOG(2) << "TraceActivity outside of profiling window: " << act.name()
        << " (" << act.timestamp() << " < " << captureWindowStartTime_ << " or "
        << (act.timestamp() + act.duration()) << " > " << captureWindowEndTime_;
  }
  return out_of_range;
}

template <typename T>
inline static bool checkRuntimeCbid(cnpapi_CallbackId cbid, const T& block_cbid_list) {
  for (const auto& block_cbid : block_cbid_list) {
    if (cbid == block_cbid) {
      return true;
    }
  }
  return false;
}

void CnpapiActivityProfiler::handleRuntimeActivity(
    const RuntimeRecord& activity,
    ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(
      activity.correlation_id, cpuCorrelationMap_);

  VLOG(2) << activity.correlation_id
          << ": CNPAPI_ACTIVITY_KIND_RUNTIME, cbid=" << activity.cbid
          << " tid=" << activity.thread_id;
  int32_t tid = activity.thread_id;
  const auto& it = resourceInfo_.find({processId(), tid});
  if (it != resourceInfo_.end()) {
    tid = it->second.id;
  }
  const auto& runtime_activity =
      traceBuffers_->addActivityWrapper(RuntimeActivity(activity, linked, tid));
  checkTimestampOrder(&runtime_activity);
  if (outOfRange(runtime_activity)) {
    return;
  }
  runtime_activity.log(*logger);
}

void CnpapiActivityProfiler::handleOverheadActivity(
    const OverheadRecord& activity,
    ActivityLogger* logger) {
  VLOG(2) << ": CNPAPI_ACTIVITY_KIND_OVERHEAD" << " overheadKind=" << activity.overhead_type;
  const auto& overhead_activity =
      traceBuffers_->addActivityWrapper(OverheadActivity(activity, nullptr));
  // Monitor memory overhead
  if (activity.overhead_type == CNPAPI_ACTIVITY_OVERHEAD_CNPAPI_RESOURCE) {
    resourceOverheadCount_++;
  }

  if (outOfRange(overhead_activity)) {
    return;
  }
  overhead_activity.log(*logger);
}


inline void CnpapiActivityProfiler::updateMluNetSpan(
    const ITraceActivity& mluOp) {
  if (!mluOp.linkedActivity()) {
    VLOG(0) << "Missing linked activity";
    return;
  }
  const auto& it = clientActivityTraceMap_.find(
     mluOp.linkedActivity()->correlationId());
  if (it == clientActivityTraceMap_.end()) {
    // No correlation id mapping?
    return;
  }
  TraceSpan& mlu_span = it->second->second;
  if (mluOp.timestamp() < mlu_span.startTime || mlu_span.startTime == 0) {
    mlu_span.startTime = mluOp.timestamp();
  }
  if ((mluOp.timestamp() + mluOp.duration()) > mlu_span.endTime) {
    mlu_span.endTime = mluOp.timestamp() + mluOp.duration();
  }
}

// I've observed occasional broken timestamps attached to MLU events...
void CnpapiActivityProfiler::checkTimestampOrder(const ITraceActivity* act1) {
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
  if (act2->type() == ActivityType::MLU_RUNTIME) {
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

inline void CnpapiActivityProfiler::handleMluActivity(
    const ITraceActivity& act,
    ActivityLogger* logger) {
  if (outOfRange(act)) {
    return;
  }
  checkTimestampOrder(&act);
  VLOG(2) << act.correlationId() << ": "
          << act.name();
  recordStream(act.deviceId(), act.resourceId(), "");
  act.log(*logger);
  updateMluNetSpan(act);
  if (derivedConfig_->profileActivityTypes().count(ActivityType::MLU_USER_ANNOTATION)) {
    const auto& it = userCorrelationMap_.find(act.correlationId());
    if (it != userCorrelationMap_.end()) {
      const auto& it2 = activityMap_.find(it->second);
      if (it2 != activityMap_.end()) {
        recordStream(act.deviceId(), act.resourceId(), "context");
        mluUserEventMap_.insertOrExtendEvent(*it2->second, act);
      }
    }
  }
}

const ITraceActivity* CnpapiActivityProfiler::linkedActivity(
    int32_t correlation_id,
    const std::unordered_map<int64_t, int64_t>& correlationMap) {
  const auto& it = correlationMap.find(correlation_id);
  if (it != correlationMap.end()) {
    const auto& it2 = activityMap_.find(it->second);
    if (it2 != activityMap_.end()) {
      return it2->second;
    }
  }
  return nullptr;
}

template <class T>
inline void CnpapiActivityProfiler::handleMluActivity(
    const T& act, std::vector<CnpapiPmuData>* pmu_data,
    ActivityLogger* logger) {
  const ITraceActivity* linked = linkedActivity(
      act.correlation_id, cpuCorrelationMap_);

  const auto& mlu_activity =
      traceBuffers_->addActivityWrapper(MluActivity<T>(act, linked, pmu_data));
  handleMluActivity(mlu_activity, logger);
}

#endif // HAS_CNPAPI

void CnpapiActivityProfiler::configureChildProfilers() {
  // If child profilers are enabled create profiler sessions
  int64_t start_time_ms = duration_cast<milliseconds>(
    derivedConfig_->profileStartTime().time_since_epoch()).count();
  for (auto& profiler: profilers_) {
    LOG(INFO) << "Running child profiler " << profiler->name() << " for "
            << derivedConfig_->profileDuration().count() << " ms";
    auto session = profiler->configure(
        start_time_ms,
        derivedConfig_->profileDuration().count(),
        derivedConfig_->profileActivityTypes(),
        *config_
    );
    if (session) {
      LOG(INFO) << "Running child profiler " << profiler->name() << " for "
                << derivedConfig_->profileDuration().count() << " ms";
      sessions_.push_back(std::move(session));
    }
  }
}

void CnpapiActivityProfiler::configure(
    const Config& config,
    const time_point<system_clock>& now) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (isActive()) {
    LOG(WARNING) << "CnpapiActivityProfiler already busy, terminating";
    return;
  }

  config_ = config.clone();

  // Ensure we're starting in a clean state
  resetTraceData();

#if !USE_GOOGLE_LOG
  // Add a LoggerObserverCollector to collect all logs during the trace.
  loggerCollectorMetadata_ = std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(loggerCollectorMetadata_.get());
#endif // !USE_GOOGLE_LOG

  derivedConfig_.reset();
  derivedConfig_ = std::make_unique<ConfigDerivedState>(*config_);

  // Check if now is a valid time to start.
  if (!derivedConfig_->canStart(now)) {
    return;
  }

  if (LOG_IS_ON(INFO)) {
    config_->printActivityProfilerConfig(LIBKINETO_DBG_STREAM);
  }
  if (!cpuOnly_ && !libkineto::api().client()) {
    if (derivedConfig_->isProfilingByIteration()) {
      LOG(INFO) << "MLU-only tracing for "
                << config_->activitiesRunIterations() << " iterations";
    } else {
      LOG(INFO) << "MLU-only tracing for "
                << config_->activitiesDuration().count() << "ms";
    }
  }

  // Set useful metadata into the logger.
  LOGGER_OBSERVER_SET_TRACE_DURATION_MS(config_->activitiesDuration().count());
  if (!config_->requestTraceID().empty()) {
    LOGGER_OBSERVER_SET_TRACE_ID(config_->requestTraceID());
  }
  if (!config_->requestGroupTraceID().empty()) {
    LOGGER_OBSERVER_SET_GROUP_TRACE_ID(config_->requestGroupTraceID());
  }
  LOGGER_OBSERVER_ADD_DESTINATION(config_->activitiesLogUrl());

#if defined(HAS_CNPAPI)
  if (!cpuOnly_) {
    // Enabling CUPTI activity tracing incurs a larger perf hit at first,
    // presumably because structures are allocated and initialized, callbacks
    // are activated etc. After a while the overhead decreases and stabilizes.
    // It's therefore useful to perform some warmup before starting recording.
    LOG(INFO) << "Enabling MLU tracing";
    cnpapi_.setMaxBufferSize(config_->activitiesMaxMluBufferSize());

    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }
    cnpapi_.enableCnpapiActivities(config_->selectedActivityTypes());
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }
#endif // HAS_CNPAPI

  if (profilers_.size() > 0) {
    configureChildProfilers();
  }

  if (libkineto::api().client()) {
    libkineto::api().client()->warmup(config_->isOpInputsCollectionEnabled());
  }

  if (derivedConfig_->isProfilingByIteration()) {
    LOG(INFO) << "Tracing starting on iteration = "
              << derivedConfig_->profileStartIteration();
    LOG(INFO) << "Tracing will end on iteration = "
              << derivedConfig_->profileEndIteration();
  } else {
    LOG(INFO) << "Tracing starting in "
              << duration_cast<seconds>(
                   derivedConfig_->profileStartTime() - now).count() << "s";
    LOG(INFO) << "Tracing will end in "
              << duration_cast<seconds>(
                   derivedConfig_->profileEndTime() - now).count() << "s";
  }

  traceBuffers_ = std::make_unique<ActivityBuffers>();
  captureWindowStartTime_ = captureWindowEndTime_ = 0;
  currentRunloopState_ = RunloopState::Warmup;
}

void CnpapiActivityProfiler::startTraceInternal(const time_point<system_clock>& now) {
  captureWindowStartTime_ = libkineto::timeSinceEpoch(now);
  VLOG(0) << "Warmup -> CollectTrace";
  for (auto& session : sessions_){
    LOG(INFO) << "Starting child profiler session";
    session->start();
  }
  currentRunloopState_ = RunloopState::CollectTrace;
}

void CnpapiActivityProfiler::stopTraceInternal(const time_point<system_clock>& now) {
  captureWindowEndTime_ = libkineto::timeSinceEpoch(now);
#if defined(HAS_CNPAPI)
  if (!cpuOnly_) {
    time_point<system_clock> timestamp;
    if (VLOG_IS_ON(1)) {
      timestamp = system_clock::now();
    }
#ifdef HAS_CNPAPI
    cnpapi_.disableCnpapiActivities(derivedConfig_->profileActivityTypes());
#else
    cnpapi_.disableActivities(derivedConfig_->profileActivityTypes());
#endif
    if (VLOG_IS_ON(1)) {
      auto t2 = system_clock::now();
      addOverheadSample(
          setupOverhead_, duration_cast<microseconds>(t2 - timestamp).count());
    }
  }
#endif // HAS_CNPAPI

  if (currentRunloopState_ == RunloopState::CollectTrace) {
    VLOG(0) << "CollectTrace -> ProcessTrace";
  } else {
    LOG(WARNING) << "Called stopTrace with state == " <<
        static_cast<std::underlying_type<RunloopState>::type>(
            currentRunloopState_.load());
  }
  for (auto& session : sessions_){
    LOG(INFO) << "Stopping child profiler session";
    session->stop();
  }
  currentRunloopState_ = RunloopState::ProcessTrace;
}

void CnpapiActivityProfiler::resetInternal() {
  resetTraceData();
  currentRunloopState_ = RunloopState::WaitForRequest;
}

const time_point<system_clock> CnpapiActivityProfiler::performRunLoopStep(
    const time_point<system_clock>& now,
    const time_point<system_clock>& nextWakeupTime,
    int64_t currentIter) {
  auto new_wakeup_time = nextWakeupTime;
  bool warmup_done = false, collection_done = false;

  VLOG_IF(1, currentIter >= 0) << "Run loop on application step(), iteration = "
    << currentIter;

  switch (currentRunloopState_) {
    case RunloopState::WaitForRequest:
      VLOG(1) << "State: WaitForRequest";
      // Nothing to do
      break;

    case RunloopState::Warmup:
      VLOG(1) << "State: Warmup";
      warmup_done = derivedConfig_->isWarmupDone(now, currentIter);
#if defined(HAS_CNPAPI)
      // Flushing can take a while so avoid doing it close to the start time
      if (!cpuOnly_ && currentIter < 0 &&
          (derivedConfig_->isProfilingByIteration() ||
           nextWakeupTime < derivedConfig_->profileStartTime())) {
        cnpapi_.clearActivities();
      }

      if (cnpapi_.stopCollection) {
        // Go to process trace to clear any outstanding buffers etc
        std::lock_guard<std::mutex> guard(mutex_);
        stopTraceInternal(now);
        resetInternal();
        VLOG(0) << "Warmup -> WaitForRequest";
        break;
      }
#endif // HAS_CNPAPI

      if (warmup_done) {
        UST_LOGGER_MARK_COMPLETED(kWarmUpStage);
        if (!derivedConfig_->isProfilingByIteration() &&
            (now > derivedConfig_->profileStartTime() + milliseconds(10))) {
          LOG(INFO)
              << "Tracing started "
              << duration_cast<milliseconds>(
                   now - derivedConfig_->profileStartTime()).count()
              << "ms late!";
        } else {
          LOG(INFO) << "Tracing started";
        }
        startTrace(now);
        if (libkineto::api().client()) {
          libkineto::api().client()->start();
        }
        if (nextWakeupTime > derivedConfig_->profileEndTime()) {
          new_wakeup_time = derivedConfig_->profileEndTime();
        }
      } else if (nextWakeupTime > derivedConfig_->profileStartTime()) {
        new_wakeup_time = derivedConfig_->profileStartTime();
      }

      break;

    case RunloopState::CollectTrace:
      VLOG(1) << "State: CollectTrace";
      collection_done = derivedConfig_->isCollectionDone(now, currentIter);

      if (collection_done
#if defined(HAS_CNPAPI)
          || cnpapi_.stopCollection
#endif // HAS_CNPAPI
      ){
        // Update runloop state first to prevent further updates to shared state
        LOG(INFO) << "Tracing complete.";
        VLOG_IF(1, currentIter > 0) << "This state change was invoked by application's step() call";

        // FIXME: Need to communicate reason for stopping on errors
        if (libkineto::api().client()) {
          libkineto::api().client()->stop();
        }
        std::lock_guard<std::mutex> guard(mutex_);
        stopTraceInternal(now);
        VLOG_IF(0, collection_done) << "Reached profile end time";

        UST_LOGGER_MARK_COMPLETED(kCollectionStage);
      } else if (derivedConfig_->isProfilingByIteration()) {
        // nothing to do here
      } else if (now < derivedConfig_->profileEndTime() &&
                 derivedConfig_->profileEndTime() < nextWakeupTime) {
        new_wakeup_time = derivedConfig_->profileEndTime();
      }

      break;

    case RunloopState::ProcessTrace:
      VLOG(1) << "State: ProcessTrace";
      // skip this state transition if it called from the step() api
      // of the profiler.
      // else it could lead to a race between the profiler thread and an
      // application thread calling step()
      if (currentIter >= 0) {
        return new_wakeup_time;
      }
      // FIXME: Probably want to allow interruption here
      // for quickly handling trace request via synchronous API
      std::lock_guard<std::mutex> guard(mutex_);
      processTraceInternal(*logger_);
      resetInternal();
      VLOG(0) << "ProcessTrace -> WaitForRequest";
      break;
  }

  return new_wakeup_time;
}

void CnpapiActivityProfiler::finalizeTrace(const Config& config, ActivityLogger& logger) {
  LOG(INFO) << "Traces Recorded:";
  {
    for (const auto& it : iterationCountMap_) {
      LOG(INFO) << it.first << ": " << it.second << " iterations";
    }
    iterationCountMap_.clear();
  }

  // Process names
  int32_t pid = processId();
  string process_name = processName(pid);
  if (!process_name.empty()) {
    logger.handleDeviceInfo(
        {pid, process_name, "CPU"}, captureWindowStartTime_);
    if (!cpuOnly_) {
      // MLU events use device id as pid (0-7).
      constexpr int kMaxMluCount = 8;
      for (int mlu = 0; mlu < kMaxMluCount; mlu++) {
        logger.handleDeviceInfo(
            {mlu, process_name, fmt::format("MLU {}", mlu)},
            captureWindowStartTime_);
      }
    }
  }

  // Thread & stream info
  for (auto pair : resourceInfo_) {
    const auto& resource = pair.second;
    logger.handleResourceInfo(resource, captureWindowStartTime_);
  }

  for (const auto& iterations : traceSpans_) {
    for (const auto& span_pair : iterations.second) {
      const TraceSpan& mlu_span = span_pair.second;
      if (mlu_span.opCount > 0) {
        logger.handleTraceSpan(mlu_span);
      }
    }
  }

  // Overhead info
  overheadInfo_.push_back(ActivityLogger::OverheadInfo("CNPAPI Overhead"));
  for(const auto& info : overheadInfo_) {
    logger.handleOverheadInfo(info, captureWindowStartTime_);
  }

  mluUserEventMap_.logEvents(&logger);

  for (auto& session : sessions_){
    auto trace_buffer = session->getTraceBuffer();
    traceBuffers_->cpu.push_back(std::move(trace_buffer));
  }

#if !USE_GOOGLE_LOG
  // Save logs from LoggerCollector objects into Trace metadata.
  auto LoggerMD = loggerCollectorMetadata_->extractCollectorMetadata();
  std::unordered_map<std::string, std::vector<std::string>> LoggerMDString;
  for (auto& md : LoggerMD) {
    LoggerMDString[toString(md.first)] = md.second;
  }
#endif // !USE_GOOGLE_LOG

  logger.finalizeTrace(config, std::move(traceBuffers_), captureWindowEndTime_, LoggerMDString);
}

void CnpapiActivityProfiler::resetTraceData() {
#if defined(HAS_CNPAPI)
  if (!cpuOnly_) {
    cnpapi_.clearActivities();
  }
#endif // HAS_CNPAPI
  activityMap_.clear();
  cpuCorrelationMap_.clear();
  correlatedMluActivities_.clear();
  mluUserEventMap_.clear();
  traceSpans_.clear();
  clientActivityTraceMap_.clear();
  traceBuffers_ = nullptr;
  metadata_.clear();
  sessions_.clear();
  resourceOverheadCount_ = 0;
#if !USE_GOOGLE_LOG
  Logger::removeLoggerObserver(loggerCollectorMetadata_.get());
#endif // !USE_GOOGLE_LOG
}


} // namespace KINETO_NAMESPACE
