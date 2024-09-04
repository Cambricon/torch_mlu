// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cnpapi.h>
#include <fmt/format.h>

#include <chrono>
#include <sstream>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ITraceActivity.h"
#include "GenericTraceActivity.h"
#include "CnpapiActivityPlatform.h"
#include "GenericTraceActivity.h"
#include "ThreadUtil.h"
#include "CnpapiRecord.h"
#include "CnpapiResourceApi.h"
#include "CnpapiPmuApi.h"
#include "cnpapi_strings.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

// These classes wrap the various CNPAPI activity types
// into subclasses of ITraceActivity so that they can all be accessed
// using the ITraceActivity interface and logged via ActivityLogger.

// Abstract base class, templated on Cnpapi activity type
template<class T>
struct CnpapiActivity : public ITraceActivity {
  explicit CnpapiActivity(const T& activity, const ITraceActivity* linked)
      : activity_(activity), linked_(linked) {}
  int64_t timestamp() const override {
    return unixEpochTimestamp(activity_.start);
  }
  int64_t duration() const override {
    return activity_.end - activity_.start;
  }
  // TODO(T107507796): Deprecate ITraceActivity
  int64_t correlationId() const override {return 0;}
  int32_t getThreadId() const override {return 0;}
  const ITraceActivity* linkedActivity() const override {return linked_;}
  int flowType() const override {return kLinkAsyncCpuMlu;}
  int flowId() const override {return correlationId();}
  const T& raw() const {return activity_;}
  const TraceSpan* traceSpan() const override {return nullptr;}

 protected:
  const T& activity_;
  const ITraceActivity* linked_{nullptr};
};

// RuntimeRecord - MLU runtime activities
struct RuntimeActivity : public CnpapiActivity<RuntimeRecord> {
  explicit RuntimeActivity(
      const RuntimeRecord& activity,
      const ITraceActivity* linked,
      int32_t threadId)
      : CnpapiActivity(activity, linked), threadId_(threadId) {}
  int64_t correlationId() const override {return activity_.correlation_id;}
  int64_t deviceId() const override {return processId();}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::MLU_RUNTIME;}
  bool flowStart() const override;
  const std::string name() const override {return runtimeCbidName(activity_.type, activity_.cbid);}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// RuntimeRecord - MLU runtime activities
struct OverheadActivity : public CnpapiActivity<OverheadRecord> {
  explicit OverheadActivity(
      const OverheadRecord& activity,
      const ITraceActivity* linked,
      int32_t threadId=0)
      : CnpapiActivity(activity, linked), threadId_(threadId) {}

  int64_t timestamp() const override {
    return unixEpochTimestamp(activity_.start);
  }
  int64_t duration() const override {
    return activity_.end - activity_.start;
  }
  // TODO: Update this with PID ordering
  int64_t deviceId() const override {return -1;}
  int64_t resourceId() const override {return threadId_;}
  ActivityType type() const override {return ActivityType::OVERHEAD;}
  bool flowStart() const override;
  const std::string name() const override {return overheadKindString(activity_.overhead_type);}
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;

 private:
  const int32_t threadId_;
};

// Base class for MLU activities.
// Can also be instantiated directly.
template<class T>
struct MluActivity : public CnpapiActivity<T> {
  explicit MluActivity(const T& activity, const ITraceActivity* linked,
                       std::vector<CnpapiPmuData>* pmu_data)
      : CnpapiActivity<T>(activity, linked), pmuData_(pmu_data) {}
  int64_t correlationId() const override {return raw().correlation_id;}
  int64_t deviceId() const override {return raw().device_id;}
  int64_t resourceId() const override {return raw().queue_id;}
  ActivityType type() const override;
  bool flowStart() const override {return false;}
  const std::string name() const override;
  void log(ActivityLogger& logger) const override;
  const std::string metadataJson() const override;
  const T& raw() const {return CnpapiActivity<T>::raw();}
 private:
  const std::string appendPmuMetadataJson() const {
    if (pmuData_ == nullptr) {
      return "";
    }
    std::ostringstream fmt_pmu_data;
    // need 6 spaces for indentation to align with other metadata
    fmt_pmu_data << ",\n      \"pmus\": {\n";
    bool is_first = true;
    for (const auto& pmu : *pmuData_) {
      if (!is_first) {
        fmt_pmu_data << ",\n";
      }
      fmt_pmu_data << "        \"" << pmu.counter_name << "\": " << pmu.value;
      is_first = false;
    }
    fmt_pmu_data << "\n      }";
    return fmt_pmu_data.str();
  }
  const std::string appendTasktopoOpNameMetadataJson() const;
  std::vector<CnpapiPmuData>* pmuData_;
};

} // namespace KINETO_NAMESPACE
