/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>

#include <libkineto.h>
#include <IActivityProfiler.h>

#include "CnpapiActivityApi.h"
#include "Config.h"

namespace libkineto_mlu {

using libkineto::Config;
using libkineto::TraceStatus;
using libkineto::ActivityLogger;
using libkineto::DeviceInfo;
using libkineto::ResourceInfo;
using libkineto::CpuTraceBuffer;
using libkineto::IActivityProfilerSession;
using libkineto::IActivityProfiler;
using libkineto::ActivityType;
using libkineto::GenericTraceActivity;
using libkineto::ITraceActivity;
using libkineto::getLinkedActivityCallback;

class CnpapiActivityProfilerSession : public IActivityProfilerSession {

  public:
    CnpapiActivityProfilerSession(const Config& config, CnpapiActivityApi& cnpapi);
    ~CnpapiActivityProfilerSession() override {}

    // start the trace collection synchronously
    // Nothing to do for cnpapi
    void start() override {};

    // stop the trace collection synchronously
    void stop() override;

    TraceStatus status() {
      return status_;
    }

    // returns errors with this trace
    std::vector<std::string> errors() override;

    // processes trace activities using logger
    void processTrace(ActivityLogger& logger) override {}
    void processTrace(ActivityLogger& logger,
      getLinkedActivityCallback getLinkedActivity,
      int64_t startTime, int64_t endTime) override;

    // returns device info used in this trace, could be nullptr
    std::unique_ptr<DeviceInfo> getDeviceInfo() override;

    // returns resource info used in this trace, could be empty
    std::vector<ResourceInfo> getResourceInfos() override;

    // release ownership of the trace events and metadata
    std::unique_ptr<CpuTraceBuffer> getTraceBuffer() override {
      return std::make_unique<CpuTraceBuffer>(std::move(traceBuffer_));
    }

    void pushCorrelationId(uint64_t id) override;
    void popCorrelationId() override;

    void pushUserCorrelationId(uint64_t id) override;
    void popUserCorrelationId() override;

  protected:
    TraceStatus status_ = TraceStatus::READY;
  
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

    // mlu runtime id -> pytorch op id
    // CNPAPI provides a mechanism for correlating Mlu events to arbitrary
    // external events, e.g.operator activities from PyTorch.
    std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
    // MLU runtime <-> MLU Activity
    std::unordered_map<int64_t, const ITraceActivity*> correlatedMluActivities_;
    std::unordered_map<int64_t, int64_t> userCorrelationMap_;

    // Buffers where mlu activities is stored
    CpuTraceBuffer traceBuffer_;

    const GenericTraceActivity* addActivityWrapper(const GenericTraceActivity& act) {
      traceBuffer_.emplace_activity(act);
      return traceBuffer_.activities.back().get();
    }

    // data structure to collect cnpapiActivityFlushAll() latency overhead
    struct profilerOverhead {
      int64_t overhead;
      int cntr;
    };

    void logMluVersions();

    // Create resource names for streams
    inline void recordStream(int device, int id, const char* postfix) {
      if (resourceInfo_.find({device, id}) == resourceInfo_.end()) {
        resourceInfo_.emplace(
            std::make_pair(device, id),
            ResourceInfo(device, id, id, fmt::format(
                    "stream {} {}", id, postfix)));
      }
    }

    const ITraceActivity* linkedActivity(
        int32_t correlationId,
        const std::unordered_map<int64_t, int64_t>& correlationMap);

    // Process specific MLU activity types
    bool outOfRange(const ITraceActivity* act);
    void handleCorrelationActivity(
        const ExternalCorrelationRecord& correlation);
    void handleRuntimeActivity(
        const RuntimeRecord& activity, ActivityLogger* logger);
    void handleOverheadActivity(
        const OverheadRecord& activity, ActivityLogger* logger);
    void handleMluActivity(const GenericTraceActivity* act,
        ActivityLogger* logger);
    template <class T>
    void handleMluActivity(const T& act, ActivityLogger* logger);

    void resetTraceData();
    void addOverheadSample(profilerOverhead& counter, int64_t overhead) {
      counter.overhead += overhead;
      counter.cntr++;
    }

    inline void updateMluNetSpan(const GenericTraceActivity* mlu_act);
    void checkTimestampOrder(const ITraceActivity* act1);

    // Logger used during trace processing
    ActivityLogger* logger_;

    // Calls to CNPAPI is encapsulated behind this interface
    CnpapiActivityApi& cnpapi_;

    // Cache thread names and system thread ids for pthread ids,
    // and stream ids for MLU streams
    std::map<
        std::pair<int64_t, int64_t>, ResourceInfo> resourceInfo_;

    // the overhead to flush the activity buffer
    profilerOverhead flushOverhead_;
    // the overhead to enable/disable activity tracking
    profilerOverhead setupOverhead_;

    std::set<ActivityType> selectedActivityTypes_;

    // Keep track of the start time and end time for the trace collected.
    // External threads using startTrace need to manually stopTrace. Part of the mock tests.
    // All MLU events before this time will be removed
    int64_t captureWindowStartTime_{0};
    // Similarly, all MLU API events after the last net event will be removed
    int64_t captureWindowEndTime_{0};

    // Use to get linked cpu activity.
    std::function<const ITraceActivity*(int32_t)> getLinkedActivity_;

    // Buffers where cnpapi trace data is stored
    std::unique_ptr<CnpapiRecordSet> cnpapiBuffers_;

    // Number of memory overhead events encountered during the session
    uint32_t resourceOverheadCount_; 
};


class CnpapiActivityProfiler : public IActivityProfiler {

  public:
    CnpapiActivityProfiler();
    ~CnpapiActivityProfiler() override {}

    // name of profiler
    const std::string& name() const override;

    // returns activity types this profiler supports
    const std::set<ActivityType>& availableActivities() const override;

    // Calls prepare() on registered tracer providers passing in the relevant
    // activity types. Returns a profiler session handle
    std::unique_ptr<IActivityProfilerSession> configure(
        const std::set<ActivityType>& activity_types,
        const Config& config) override;

    // asynchronous version of the above with future timestamp and duration.
    std::unique_ptr<IActivityProfilerSession> configure(
        int64_t ts_ms,
        int64_t duration_ms,
        const std::set<ActivityType>& activity_types,
        const Config& config) override;
};

struct CnpapiActivityProfilerInit {
  CnpapiActivityProfilerInit();
};

}  // libkineto_mlu
