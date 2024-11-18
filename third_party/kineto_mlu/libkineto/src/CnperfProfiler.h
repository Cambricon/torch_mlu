#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude

#include <cnperf_api.h>
#include "CnperfApi.h"
#include "CnperfActivity.h"
#include "CnperfPmuApi.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"
#include "libkineto.h"
#include "output_base.h"
#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "LoggerCollector.h"

namespace KINETO_NAMESPACE {

class Config;
class CnperfApi;

// This struct is a derived snapshot of the Config. And should not
// be mutable after construction.
struct ConfigDerivedState final {
  ConfigDerivedState() = delete;
  ConfigDerivedState(const Config&);

  // Calculate if starting is valid.
  bool canStart(
    const std::chrono::time_point<std::chrono::system_clock>& now) const;

  // TODO: consider using union since only 1 arg is used.
  bool isWarmupDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const;

  bool isCollectionDone(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      int64_t currentIter) const;

  // Set and Get Functions below.
  const std::set<ActivityType>& profileActivityTypes() const {
    return profileActivityTypes_;
  }

  const std::chrono::time_point<std::chrono::system_clock>
  profileStartTime() const {
    return profileStartTime_;
  }

  const std::chrono::time_point<std::chrono::system_clock>
  profileEndTime() const {
    return profileEndTime_;
  }

  const std::chrono::milliseconds
  profileDuration() const {
    return profileDuration_;
  }

  int64_t profileStartIteration() const { return profileStartIter_; }
  int64_t profileEndIteration() const { return profileEndIter_; }
  bool isProfilingByIteration() const { return profilingByIter_; }

 private:
  std::set<ActivityType> profileActivityTypes_;
  // Start and end time used for triggering and stopping profiling
  std::chrono::time_point<std::chrono::system_clock> profileStartTime_;
  std::chrono::time_point<std::chrono::system_clock> profileEndTime_;
  std::chrono::milliseconds profileDuration_;
  std::chrono::seconds profileWarmupDuration_;
  int64_t profileStartIter_ {-1};
  int64_t profileEndIter_ {-1};
  bool profilingByIter_ {false};
};

class CnperfProfiler {
 public:
  CnperfProfiler(CnperfApi& cnperf, bool cpuOnly);
  CnperfProfiler(const CnperfProfiler&) = delete;
  CnperfProfiler& operator=(const CnperfProfiler&) = delete;

  bool isActive() const {
    return currentRunloopState_ != RunloopState::WaitForRequest;
  }

  // Invoke at a regular interval to perform profiling activities.
  // When not active, an interval of 1-5 seconds is probably fine,
  // depending on required warm-up time and delayed start time.
  // When active, it's a good idea to invoke more frequently to stay below
  // memory usage limit (ACTIVITIES_MAX_MLU_BUFFER_SIZE_MB) during warmup.
  const std::chrono::time_point<std::chrono::system_clock> performRunLoopStep(
      const std::chrono::time_point<std::chrono::system_clock>& now,
      const std::chrono::time_point<std::chrono::system_clock>& nextWakeupTime,
      int64_t currentIter = -1);

  // Used for async requests
  void setLogger(ActivityLogger* logger) {
    logger_ = logger;
  }

  // Synchronous control API
  void startTrace(
      const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    startTraceInternal(now);
  }

  void stopTrace(const std::chrono::time_point<std::chrono::system_clock>& now) {
    std::lock_guard<std::mutex> guard(mutex_);
    stopTraceInternal(now);
  }

  // Process CPU and MLU traces
  void processTrace(ActivityLogger& logger) {
    std::lock_guard<std::mutex> guard(mutex_);
    processTraceInternal(logger);
  }

  void reset() {
    std::lock_guard<std::mutex> guard(mutex_);
    resetInternal();
  }

  // Set up profiler as specified in config.
  void configure(
      const Config& config,
      const std::chrono::time_point<std::chrono::system_clock>& now);

  // Registered with client API to pass CPU trace events over
  void transferCpuTrace(
      std::unique_ptr<libkineto::CpuTraceBuffer> cpuTrace);

  const Config& config() {
    return *config_;
  }

  inline void recordThreadInfo() {
    int32_t sysTid = systemThreadId();
    // Note we're using the lower 32 bits of the (opaque) pthread id
    // as key, because that's what CNPERF records.
    int32_t tid = threadId();
    int32_t pid = processId();
    std::lock_guard<std::mutex> guard(mutex_);
    recordThreadInfo(sysTid, tid, pid);
  }

  // T107508020: We can deprecate the recordThreadInfo(void) once we optimized profiler_kineto
  void recordThreadInfo(int32_t sysTid, int32_t tid, int32_t pid) {
    if (resourceInfo_.find({pid, tid}) == resourceInfo_.end()) {
      resourceInfo_.emplace(
          std::make_pair(pid, tid),
          ActivityLogger::ResourceInfo(
              pid,
              sysTid,
              sysTid, // sortindex
              fmt::format("thread {} ({})", sysTid, getThreadName())));
    }
  }

  void addMetadata(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> guard(mutex_);
    metadata_[key] = value;
  }

 protected:

  using CpuMluSpanPair = std::pair<TraceSpan, TraceSpan>;
  static const CpuMluSpanPair& defaultTraceSpan();

 private:

  // Map of mlu activities to user defined events
  class MluUserEventMap {
   public:
    // Insert a user defined event which maps to the mlu trace activity.
    // If the user defined event mapping already exists this will update the
    // mlu side span to include the span of mluTraceActivity.
    void insertOrExtendEvent(const ITraceActivity& cpuTraceActivity,
      const ITraceActivity& mluTraceActivity);
    // Log out the events to the logger
    void logEvents(ActivityLogger *logger);

    void clear() {
      streamSpanMap_.clear();
    }

   private:
    // device id and stream name
    using StreamKey = std::pair<int64_t, int64_t>;

    // map of correlation id to TraceSpan
    using CorrelationSpanMap =
        std::unordered_map<int64_t, GenericTraceActivity>;
    std::map<StreamKey, CorrelationSpanMap> streamSpanMap_;
  };

  MluUserEventMap mluUserEventMap_;
  // id -> activity*
  std::unordered_map<int64_t, const ITraceActivity*> activityMap_;

  // MLU runtime <-> MLU Activity
  std::unordered_map<int64_t, const ITraceActivity*>
      correlatedMluActivities_;

  void logMluVersions();

  void startTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void stopTraceInternal(
      const std::chrono::time_point<std::chrono::system_clock>& now);

  void processTraceInternal(ActivityLogger& logger);

  void resetInternal();

  void finalizeTrace(const Config& config, ActivityLogger& logger);

  // Process a single CPU trace
  void processCpuTrace(
      libkineto::CpuTraceBuffer& cpuTrace,
      ActivityLogger& logger);

  // Create resource names for streams
  inline void recordStream(int device, int id, const char* postfix) {
    if (resourceInfo_.find({device, id}) == resourceInfo_.end()) {
      resourceInfo_.emplace(
          std::make_pair(device, id),
          ActivityLogger::ResourceInfo(
              device, id, id, fmt::format(
                  "stream {} {}", id, postfix)));
    }
  }

  inline void recordCnpxInfo(int rank, int tid) {
    if (resourceInfo_.find({rank, tid}) == resourceInfo_.end()) {
      resourceInfo_.emplace(
          std::make_pair(rank, tid),
          ActivityLogger::ResourceInfo(
              rank,
              tid,
              tid, // sortindex
              fmt::format("cnpx (thread {})", tid)));
    }
  }


  // Record client trace span for subsequent lookups from activities
  // Also creates a corresponding MLU-side span.
  CpuMluSpanPair& recordTraceSpan(TraceSpan& span, int mluOpCount);

  // Returns true if net name is to be tracked for a specified number of
  // iterations.
  bool iterationTargetMatch(libkineto::CpuTraceBuffer& trace);

  // net name to id
  int netId(const std::string& netName);

  const ITraceActivity* linkedActivity(int32_t correlationId);

  // Process specific MLU activity types
  void updateMluNetSpan(const ITraceActivity& mluOp);
  bool outOfRange(const ITraceActivity& act);
  void handleRuntimeActivity(
      const RuntimeRecord& activity,
      const std::unique_ptr<std::unordered_set<uint64_t>>& device_task_corrids,
      ActivityLogger* logger);
  void handleMluActivity(const DeviceTaskRecord& act,
      std::vector<CnperfPmuData>* pmu_data,
      ActivityLogger* logger);
  void handleCommunicationActivity(
      const CommunicationRecord& record,
      ActivityLogger* logger);

  void resetTraceData();

  void checkTimestampOrder(const ITraceActivity* act1);

  // On-demand Request Config (should not be modified)
  // TODO: remove this config_, dependency needs to be removed from finalizeTrace.
  std::unique_ptr<const Config> config_;

  // Resolved details about the config and states are stored here.
  std::unique_ptr<ConfigDerivedState> derivedConfig_;

  // Logger used during trace processing
  ActivityLogger* logger_;

  CnperfApi& cnperf_;

  enum class RunloopState {
    WaitForRequest,
    Warmup,
    CollectTrace,
    ProcessTrace
  };

  // All recorded trace spans, both CPU and MLU
  // Trace Id -> list of iterations.
  // Using map of lists for the iterator semantics, since we are recording
  // pointers to the elements in this structure.
  std::map<std::string, std::list<CpuMluSpanPair>> traceSpans_;

  // Maintain a map of client trace activity to trace span.
  // Maps correlation id -> TraceSpan* held by traceSpans_.
  using ActivityTraceMap = std::unordered_map<int64_t, CpuMluSpanPair*>;
  ActivityTraceMap clientActivityTraceMap_;

  // Cache thread names and system thread ids for pthread ids,
  // and stream ids for MLU streams
  std::map<
      std::pair<int64_t, int64_t>,
      ActivityLogger::ResourceInfo> resourceInfo_;

  std::vector<ActivityLogger::OverheadInfo> overheadInfo_;

  bool cpuOnly_{false};

  // ***************************************************************************
  // Below state is shared with external threads.
  // These need to either be atomic, accessed under lock or only used
  // by external threads in separate runloop phases from the profiler thread.
  // ***************************************************************************

  // Mutex to protect non-atomic access to below state
  std::mutex mutex_;

  // Runloop phase
  std::atomic<RunloopState> currentRunloopState_{RunloopState::WaitForRequest};

  // Keep track of the start time and end time for the trace collected.
  // External threads using startTrace need to manually stopTrace. Part of the mock tests.
  // All MLU events before this time will be removed
  int64_t captureWindowStartTime_{0};
  // Similarly, all MLU API events after the last net event will be removed
  int64_t captureWindowEndTime_{0};

  // span name -> iteration count
  std::map<std::string, int> iterationCountMap_;

  // Buffers where trace data is stored
  std::unique_ptr<ActivityBuffers> traceBuffers_;

  // Trace metadata
  std::unordered_map<std::string, std::string> metadata_;

  // Number of memory overhead events encountered during the session
  uint32_t resourceOverheadCount_; 

  // LoggerCollector to collect all LOGs during the trace
#if !USE_GOOGLE_LOG
  std::unique_ptr<LoggerCollector> loggerCollectorMetadata_;
#endif // !USE_GOOGLE_LOG
};

} // namespace KINETO_NAMESPACE
