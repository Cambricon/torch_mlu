// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "output_json.h"

#include <fmt/format.h>
#include <fstream>
#include <time.h>
#include <map>

#include "Config.h"
#ifdef HAS_CNPAPI
#include "MluDeviceProperties.h"
#endif // HAS_CNPAPI
#include "Demangle.h"
#include "TraceSpan.h"

#include "Logger.h"

using std::endl;
using namespace libkineto;

namespace KINETO_NAMESPACE {

static constexpr int kSchemaVersion = 1;
static constexpr char kFlowStart = 's';
static constexpr char kFlowEnd = 'f';

#ifdef __linux__
static constexpr char kDefaultLogFileFmt[] =
    "/tmp/libkineto_activities_{}.json";
#else
static constexpr char kDefaultLogFileFmt[] = "libkineto_activities_{}.json";
#endif

ChromeTraceBaseTime& ChromeTraceBaseTime::singleton() {
  static ChromeTraceBaseTime instance;
  return instance;
}

// The 'ts' field written into the json file has 19 significant digits,
// while a double can only represent 15-16 digits. By using relative time,
// other applications can accurately read the 'ts' field as a double.
// Use the program loading time as the baseline time.
inline int64_t transToRelativeTime(int64_t time) {
  // Sometimes after converting to relative time, it can be a few nanoseconds negative.
  // Since Chrome trace and json processing will throw a parser error, guard this.
  int64_t res = time - ChromeTraceBaseTime::singleton().get();
  if (res < 0) {
    return 0;
  }
  return res;
}

std::string& ChromeTraceLogger::sanitizeStrForJSON(std::string& value) {
// Replace all backslashes with forward slash because Windows paths causing JSONDecodeError.
#ifdef _WIN32
  std::replace(value.begin(), value.end(), '\\', '/');
#endif
  return value;
}

void ChromeTraceLogger::metadataToJSON(
    const std::unordered_map<std::string, std::string>& metadata) {
  for (const auto& kv : metadata) {
    traceOf_ << fmt::format(R"JSON(
  "{}": {},)JSON", kv.first, kv.second);
  }
}

void ChromeTraceLogger::handleTraceStart(
    const std::unordered_map<std::string, std::string>& metadata) {
  traceOf_ << fmt::format(R"JSON(
{{
  "schemaVersion": {},)JSON", kSchemaVersion);

#ifdef HAS_CNPAPI
  traceOf_ << fmt::format(R"JSON(
  "deviceProperties": [{}
  ],)JSON", devicePropertiesJson());
#endif

  metadataToJSON(metadata);
  traceOf_ << R"JSON(
  "traceEvents": [)JSON";
}

static std::string defaultFileName() {
  return fmt::format(kDefaultLogFileFmt, processId());
}

void ChromeTraceLogger::openTraceFile() {
  traceOf_.open(fileName_, std::ofstream::out | std::ofstream::trunc);
  if (!traceOf_) {
    PLOG(ERROR) << "Failed to open '" << fileName_ << "'";
  } else {
    LOG(INFO) << "Tracing to " << fileName_;
  }
}

ChromeTraceLogger::ChromeTraceLogger(const std::string& traceFileName) {
  fileName_ = traceFileName.empty() ? defaultFileName() : traceFileName;
  traceOf_.clear(std::ios_base::badbit);
  openTraceFile();
}

void ChromeTraceLogger::handleDeviceInfo(
    const DeviceInfo& info,
    uint64_t time) {
  if (!traceOf_) {
    return;
  }

  // M is for metadata
  // process_name needs a pid and a name arg
  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  time = transToRelativeTime(time);
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_labels", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": 0,
    "args": {{
      "labels": "{}"
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time/1000, time%1000, info.id,
      info.name,
      time/1000, time%1000, info.id,
      info.label,
      time/1000, time%1000, info.id,
      info.id < 8 ? info.id + 0x1000000ll : info.id);
  // clang-format on
}

void ChromeTraceLogger::handleResourceInfo(
    const ResourceInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // M is for metadata
  // thread_name needs a pid and a name arg
  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  time = transToRelativeTime(time);
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "thread_name", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": {},
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "thread_sort_index", "ph": "M", "ts": {}.{:03}, "pid": {}, "tid": {},
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time/1000, time%1000, info.deviceId, info.id,
      info.name,
      time/1000, time%1000, info.deviceId, info.id,
      info.sortIndex);
  // clang-format on
}

void ChromeTraceLogger::handleOverheadInfo(
    const OverheadInfo& info,
    int64_t time) {
  if (!traceOf_) {
    return;
  }

  // TOOD: reserve pid = -1 for overhead but we need to rethink how to scale this for
  // other metadata
  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  time = transToRelativeTime(time);
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "process_name", "ph": "M", "ts": {}.{:03}, "pid": -1, "tid": 0,
    "args": {{
      "name": "{}"
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}.{:03}, "pid": -1, "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      time/1000, time%1000,
      info.name,
      time/1000, time%1000,
      0x100000All);
  // clang-format on
}

void ChromeTraceLogger::handleTraceSpan(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }
  // see [Note: Temp Libkineto Nanosecond]
  uint64_t start = transToRelativeTime(span.startTime);
  uint64_t dur = span.endTime - span.startTime;

  // clang-format off
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "Trace", "ts": {}.{:03}, "dur": {}.{:03},
    "pid": "Spans", "tid": "{}",
    "name": "{}{} ({})",
    "args": {{
      "Op count": {}
    }}
  }},
  {{
    "name": "process_sort_index", "ph": "M", "ts": {}.{:03},
    "pid": "Spans", "tid": 0,
    "args": {{
      "sort_index": {}
    }}
  }},)JSON",
      start/1000, start%1000, dur/1000, dur%1000,
      span.name,
      span.prefix, span.name, span.iteration,
      span.opCount,
      start/1000, start%1000,
      // Large sort index to appear at the bottom
      0x20000000ll);
  // clang-format on

  addIterationMarker(span);
}

void ChromeTraceLogger::addIterationMarker(const TraceSpan& span) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  uint64_t start = transToRelativeTime(span.startTime);

  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Iteration Start: {}", "ph": "i", "s": "g",
    "pid": "Traces", "tid": "Trace {}", "ts": {}.{:03}
  }},)JSON",
      span.name,
      span.name, start/1000, start%1000);
  // clang-format on
}

void ChromeTraceLogger::handleGenericInstantEvent(
    const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }
  // see [Note: Temp Libkineto Nanosecond]
  uint64_t ts = transToRelativeTime(op.timestamp());
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "i", "cat": "{}", "s": "t", "name": "{}",
    "pid": {}, "tid": {},
    "ts": {}.{:03},
    "args": {{
      {}
    }}
  }},)JSON",
      toString(op.type()), op.name(), op.deviceId(), op.resourceId(),
      ts/1000, ts%1000, op.metadataJson());
}

void ChromeTraceLogger::handleActivity(
    const libkineto::ITraceActivity& op) {
  if (!traceOf_) {
    return;
  }

  if (op.type() == ActivityType::CPU_INSTANT_EVENT) {
    handleGenericInstantEvent(op);
    return;
  }

  int64_t ts = op.timestamp();
  int64_t duration = op.duration();
  if (op.type() ==  ActivityType::MLU_USER_ANNOTATION) {
    // The MLU user annotations start at the same time as the
    // first associated MLU op. Since they appear later
    // in the trace file, this causes a visualization issue in Chrome.
    // Make it start one us (1000 ns) earlier.
    ts-=1000;
    duration+=1500; // Still need it to end at the original point rounded up.
  }
  const std::string op_metadata = op.metadataJson();
  std::string separator = "";
  if (op_metadata.find_first_not_of(" \t\n") != std::string::npos) {
    separator = ",\n      ";
  }
  std::string span = "";
  if (op.traceSpan()) {
    span = fmt::format(R"JSON(
      "Trace name": "{}", "Trace iteration": {},)JSON",
        op.traceSpan()->name,
        op.traceSpan()->iteration);
  }
  int device = op.deviceId();
  int resource = op.resourceId();

  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  ts = transToRelativeTime(ts);
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "X", "cat": "{}", "name": "{}", "pid": {}, "tid": {},
    "ts": {}.{:03}, "dur": {}.{:03},
    "args": {{{}
      "External id": {}{}{}
    }}
  }},)JSON",
          toString(op.type()), op.name(), device, resource,
          ts/1000, ts %1000, duration/1000, duration %1000,
          // args
          span,
          op.linkedActivity() ?
          op.linkedActivity()->correlationId() :
          op.correlationId(),
          separator, op_metadata);
  // clang-format on
  if (op.flowId() > 0) {
    handleGenericLink(op);
  }
}

void ChromeTraceLogger::handleGenericActivity(
    const libkineto::GenericTraceActivity& op) {
        handleActivity(op);
}

void ChromeTraceLogger::handleGenericLink(const ITraceActivity& act) {
  static struct {
    int type;
    char longName[24];
    char shortName[16];
  } flow_names[] = {
    {kLinkFwdBwd, "forward_backward", "fwd_bwd"},
    {kLinkAsyncCpuMlu, "async_cpu_to_mlu", "async_mlu"}
  };
  for (auto& flow : flow_names) {
    if (act.flowType() == flow.type) {
      // Link the activities via flow ID in source and destination.
      // The source node must return true from flowStart()
      // and the destination node false.
      if (act.flowStart()) {
        handleLink(kFlowStart, act, act.flowId(), flow.longName, flow.shortName);
      } else {
        handleLink(kFlowEnd, act, act.flowId(), flow.longName, flow.shortName);
      }
      return;
    }
  }
  LOG(WARNING) << "Unknown flow type: " << act.flowType();
}

void ChromeTraceLogger::handleLink(
    char type,
    const ITraceActivity& e,
    int64_t id,
    const std::string& cat,
    const std::string& name) {
  if (!traceOf_) {
    return;
  }

  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  uint64_t ts = transToRelativeTime(e.timestamp());
  traceOf_ << fmt::format(R"JSON(
  {{
    "ph": "{}", "id": {}, "pid": {}, "tid": {}, "ts": {}.{:03},
    "cat": "{}", "name": "{}", "bp": "e"
  }},)JSON",
      type, id, e.deviceId(), e.resourceId(), ts/1000, ts%1000, cat, name);
  // clang-format on
}

void ChromeTraceLogger::finalizeTraceInternal(
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  if (!traceOf_) {
    LOG(ERROR) << "Failed to write to log file!";
    return;
  }
  LOG(INFO) << "Chrome Trace written to " << fileName_;
  // clang-format off
  // see [Note: Temp Libkineto Nanosecond]
  endTime = transToRelativeTime(endTime);
  traceOf_ << fmt::format(R"JSON(
  {{
    "name": "Record Window End", "ph": "i", "s": "g",
    "pid": "", "tid": "", "ts": {}.{:03}
  }}
  ],)JSON",
      endTime/1000, endTime %1000);

#if !USE_GOOGLE_LOG
  std::unordered_map<std::string, std::string> PreparedMetadata;
  for (const auto& kv : metadata) {
    // Skip empty log buckets, ex. skip ERROR if its empty.
    if (!kv.second.empty()) {
      std::string value = "[";
      // Ex. Each metadata from logger is a list of strings, expressed in JSON as
      //   "ERROR": ["Error 1", "Error 2"],
      //   "WARNING": ["Warning 1", "Warning 2", "Warning 3"],
      //   ...
      int mdv_count = kv.second.size();
      for (const auto& v : kv.second) {
        value.append("\"" + v + "\"");
        if(mdv_count > 1) {
          value.append(",");
          mdv_count--;
        }
      }
      value.append("]");
      PreparedMetadata[kv.first] = sanitizeStrForJSON(value);
    }
  }
  metadataToJSON(PreparedMetadata);
#endif // !USE_GOOGLE_LOG

  // Putting this here because the last entry MUST not end with a comma.
  traceOf_ << fmt::format(R"JSON(
  "traceName": "{}",
  "displayTimeUnit": "ms",
  "baseTimeNanoseconds": {}
}})JSON", sanitizeStrForJSON(fileName_), ChromeTraceBaseTime::singleton().get());
  // clang-format on

  traceOf_.close();
}

void ChromeTraceLogger::finalizeTrace(
    const Config& /*unused*/,
    std::unique_ptr<ActivityBuffers> /*unused*/,
    int64_t endTime,
    std::unordered_map<std::string, std::vector<std::string>>& metadata) {
  ChromeTraceLogger::finalizeTraceInternal(endTime, metadata);
  UST_LOGGER_MARK_COMPLETED(kPostProcessingStage);
}
} // namespace KINETO_NAMESPACE
