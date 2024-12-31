#include "CnperfApi.h"

#include "ApiListCommon.h"
#include "ILoggerObserver.h"
#include "Logger.h"
#include "ThreadUtil.h"

#include <assert.h>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <filesystem>
#include <unistd.h>

using namespace std::chrono;

namespace KINETO_NAMESPACE {

const size_t USER_EXTERNAL_IDS_DEFAULT_CAPACITY = 1024 * 1024;

CnperfApi& CnperfApi::singleton() {
  static CnperfApi instance;
  return instance;
}

CnperfApi::CnperfApi() {
  // define the folder path to save cnperf data
  const char* raw_data_path = std::getenv("TORCH_MLU_PROFILER_OUTPUT_PATH");
  const char* data_path =
      raw_data_path ? raw_data_path : "./TORCH_PROFILER_RAW_DATA";
  const std::filesystem::path root_path{data_path};
  const std::filesystem::path pid{std::to_string(processId())};
  const std::filesystem::path fspath = root_path / pid;
  if (std::filesystem::is_directory(fspath)) {
    int ret = access(fspath.c_str(), W_OK);
    // return 0 means ok with write permission
    if (ret != 0) {
      LOG(ERROR) << "Write permission denied for directory: [" << data_path
                 << "]";
      exit(1);
    }
  } else {
    try {
      std::filesystem::create_directories(fspath);
    } catch (std::filesystem::filesystem_error const& ex) {
      LOG(ERROR) << ex.what();
      exit(1);
    }
  }
  CNPERF_CALL(cnperfInit(0));
  CNPERF_CALL(cnperfSetBaseDir(fspath.string().c_str()));

  // Keep the user_external_ids_ lifecycle consistent with the process.
  // To obtain the tasktopo op name, it may be necessary to know
  // the user external id of the previous profile in same process.
  user_external_ids_.reserve(USER_EXTERNAL_IDS_DEFAULT_CAPACITY);

  // whether enable catching mlugraph's op or not
  enable_catching_mlugraph_op_ = [] {
    bool result = false;
    const auto env = std::getenv("TORCH_MLU_ENABLE_CATCHING_MLUGRAPH_OP");
    if (env) {
      std::string env_str = env;
      std::transform(
          env_str.begin(), env_str.end(), env_str.begin(), [](unsigned char c) {
            return std::tolower(c);
          });
      if (env_str == "true" || env_str == "1" || env_str == "on") {
        result = true;
      }
    }
    return result;
  }();
}

void CnperfApi::pushCorrelationID(int id, CorrelationFlowType type) {
  if (type == User) {
    singleton().user_external_ids_.insert(id);
  }
}

void CnperfApi::popCorrelationID(CorrelationFlowType type) {}

void CnperfApi::fillCpuActivities(
    const std::unordered_map<int64_t, const ITraceActivity*>& cpu_activities) {
  for (const auto& id_to_act : cpu_activities) {
    const auto& cpu_act = id_to_act.second;
    cnperfDataOpRange_t op_range;
    op_range.version = CNPERF_DATA_VERSION_CURRENT;
    op_range.external_correlation_id = id_to_act.first;
    // Here set param start_ts of transCPUTimeToCnperfTime to 0 because
    // of Cnperf internal will minus start_ts.
    op_range.start = transCPUTimeToCnperfTime(cpu_act->timestamp(), 0);
    op_range.end =
        transCPUTimeToCnperfTime(cpu_act->timestamp() + cpu_act->duration(), 0);
    op_range.thread_id = cpu_act->resourceId();
    // avoid using rvalue std::string here.
    auto temp_name = cpu_act->name();
    op_range.name = temp_name.c_str();
    op_range.extra = "";
    CNPERF_CALL(cnperfSessionRecordData(
        session_, CNPERF_RECORD_DATA_TYPE_OP_RANGE, &op_range));
  }
}

// static
void CnperfApi::saveExternalId(
    cnperfDataOpRange_t* data,
    std::vector<std::pair<uint64_t, uint64_t>>* time_to_external_id) {
  if (data->external_correlation_id > 0 &&
      static_cast<int64_t>(data->start) > 0) {
    time_to_external_id->emplace_back(
        std::make_pair(data->start, data->external_correlation_id));
  }
}

// static
void CnperfApi::saveTasktopoExternalOp(
    cnperfDataOpRange_t* data,
    std::vector<std::pair<uint64_t, std::pair<uint64_t, std::string>>>*
        time_to_external_op) {
  // Don't need link user_annotation to mlugraph kernels
  if (singleton().user_external_ids_.find(data->external_correlation_id) ==
      singleton().user_external_ids_.end()) {
    time_to_external_op->emplace_back(std::make_pair(
        data->start,
        std::make_pair(data->external_correlation_id, data->name)));
  }
}

// static
void CnperfApi::savePmuData(
    const cnperfDataPmuInfo_t* pmu_info,
    const cnperfDataPmuValue_t* value,
    uint64_t* corrid) {
  // -1 represents all clusters/cores' total value
  if (pmu_info->cluster_id != -1 || pmu_info->core_id != -1)
    return;
  uint64_t cast_value = 0;
  CnperfPmuData cur_pmu{pmu_info->name, uint64_t(0)};
  switch (pmu_info->type) {
    case CNPERF_DATA_PMU_VALUE_TYPE_UINT64:
      cur_pmu.value = value->u64;
      break;
    case CNPERF_DATA_PMU_VALUE_TYPE_INT64:
      cur_pmu.value = value->i64;
      break;
    case CNPERF_DATA_PMU_VALUE_TYPE_DOUBLE:
      cur_pmu.value = value->dbl;
      break;
    default:
      break;
  }
  CnperfPmuApi::singleton().recordPmuData(*corrid, cur_pmu);
}

// static
void CnperfApi::processFunction(cnperfDataFunction_t* data, void* userdata) {
  singleton().all_records_->runtime_records.emplace_back(
      data->correlation_id,
      transCnperfTimeToCPUTime(data->start, singleton().cnperf_start_ts_),
      transCnperfTimeToCPUTime(data->end, singleton().cnperf_start_ts_),
      data->thread_id,
      data->name,
      data->extra);
  // Find the external_id for each runtime api base on corrid.
  // CnperfApi will return all external apis. We only pick top
  // of stack, that 'start' is bigest.
  std::vector<std::pair<uint64_t, uint64_t>> time_to_external_id;
  // Usually the external call stack is not too deep,
  // set to 32 here to reduce the overhead of reallocating memory.
  time_to_external_id.reserve(32);
  CNPERF_CALL(cnperfParserGetOpRanges(
      singleton().parser_,
      data->correlation_id,
      (cnperfParserDataCallback)CnperfApi::saveExternalId,
      &time_to_external_id));
  if (!time_to_external_id.empty()) {
    std::pair<uint64_t, uint64_t> default_id = {0, 0};
    std::pair<uint64_t, uint64_t> user_id = {0, 0};
    for (const auto& [time, id] : time_to_external_id) {
      if (singleton().user_external_ids_.find(id) !=
              singleton().user_external_ids_.end() &&
          user_id.first < time) {
        user_id = {time, id};
      }
      if (singleton().user_external_ids_.find(id) ==
              singleton().user_external_ids_.end() &&
          default_id.first < time) {
        default_id = {time, id};
      }
    }
    if (user_id.second > 0) {
      singleton().user_correlation_map_.insert(
          {data->correlation_id, user_id.second});
    }
    if (default_id.second > 0) {
      singleton().cpu_correlation_map_.insert(
          {data->correlation_id, default_id.second});
    }
  }
}

// static
void CnperfApi::processDeviceTask(
    cnperfDataDeviceTask_t* data,
    void* userdata) {
  std::pair<uint64_t, std::string> tasktopo_external_op;
  if (singleton().enable_catching_mlugraph_op_) {
    // Time to op id and op name
    std::vector<std::pair<uint64_t, std::pair<uint64_t, std::string>>>
        time_to_external_op;
    // Usually the external call stack is not too deep,
    // set to 32 here to reduce the overhead of reallocating memory.
    time_to_external_op.reserve(32);
    CNPERF_CALL(cnperfParserGetTaskTopoNodeOpRanges(
        singleton().parser_,
        data->tasktopo_id,
        data->tasktopo_node_id,
        (cnperfParserDataCallback)CnperfApi::saveTasktopoExternalOp,
        &time_to_external_op));
    if (!time_to_external_op.empty()) {
      auto top_external_op_iter = std::max_element(
          time_to_external_op.begin(),
          time_to_external_op.end(),
          [](const auto& a, const auto& b) { return a.first < b.first; });
      if (top_external_op_iter != time_to_external_op.end()) {
        tasktopo_external_op = {
            top_external_op_iter->second.first,
            top_external_op_iter->second.second};
      }
    }
  }
  singleton().all_records_->device_task_records.emplace_back(
      data->type,
      data->process_id,
      data->correlation_id,
      transCnperfTimeToCPUTime(data->start, singleton().cnperf_start_ts_),
      transCnperfTimeToCPUTime(data->end, singleton().cnperf_start_ts_),
      data->device_id,
      data->context_id,
      data->queue_id,
      data->tasktopo_id,
      data->tasktopo_node_id,
      data->is_async,
      data->name,
      data->extra,
      tasktopo_external_op.first,
      std::move(tasktopo_external_op.second));
  singleton().device_task_correlations_->emplace(data->correlation_id);
  if (CnperfPmuApi::singleton().enabled()) {
    uint64_t cur_corrid = data->correlation_id;
    CNPERF_CALL(cnperfParserGetKernelPmuData(
        singleton().parser_,
        data->correlation_id,
        data->tasktopo_id,
        data->tasktopo_node_id,
        (cnperfParserKernelPmuDataCallback)CnperfApi::savePmuData,
        &cur_corrid));
  }
}

// static
void CnperfApi::processCommData(
    const cnperfDataOpRange_t* op_range,
    const cnperfDataCommTask_t* data,
    void* userdata) {
  if (singleton().user_external_ids_.find(op_range->external_correlation_id) !=
          singleton().user_external_ids_.end() &&
      std::string(op_range->name).find("cncl:") == 0) {
    singleton().all_records_->comm_records.emplace_back(
        op_range->thread_id,
        op_range->external_correlation_id,
        op_range->name,
        data->type,
        data->rank,
        data->clique_id,
        transCnperfTimeToCPUTime(data->start, singleton().cnperf_start_ts_),
        transCnperfTimeToCPUTime(data->end, singleton().cnperf_start_ts_),
        data->bytes,
        data->name);
  }
}

void CnperfApi::prepare() {
  CNPERF_CALL(cnperfConfigCreate(&config_));
  ApiList::getInstance().updateConfig(config_);
  CnperfPmuApi::singleton().updateConfig(config_);
  CNPERF_CALL(cnperfConfigEnable(config_));
  all_records_ = std::make_unique<CnperfRecordSet>();
  device_task_correlations_ = std::make_unique<std::unordered_set<uint64_t>>();
  // Start session in prepare() to align with native.
  CNPERF_CALL(cnperfStart(&session_));
}

void CnperfApi::start() {}

// Calling cnperfStop() in process() instead of in stop() as we need to
// fill cpu activities to cnperf before cnperfStop().
void CnperfApi::stop() {}

void CnperfApi::process(
    const std::unordered_map<int64_t, const ITraceActivity*>& cpu_activities) {
  fillCpuActivities(cpu_activities);

  CNPERF_CALL(cnperfStop());

  CNPERF_CALL(cnperfParserCreateFromSession(session_, &parser_));
  CNPERF_CALL(cnperfParserGetStartTimestamp(parser_, &cnperf_start_ts_));
  CNPERF_CALL(cnperfParserGetData(
      parser_,
      CNPERF_PARSER_DATA_TYPE_FUNCTION,
      (cnperfParserDataCallback)CnperfApi::processFunction,
      nullptr));
  CNPERF_CALL(cnperfParserGetData(
      parser_,
      CNPERF_PARSER_DATA_TYPE_DEVICE_TASK,
      (cnperfParserDataCallback)CnperfApi::processDeviceTask,
      nullptr));
  CNPERF_CALL(
      cnperfParserGetCommDataInOpRange(parser_, processCommData, nullptr));

  CNPERF_CALL(cnperfParserDestroy(parser_));
  CNPERF_CALL(cnperfSessionDestroy(session_));
  CNPERF_CALL(cnperfConfigDestroy(config_));
}

void CnperfApi::clearTraceData() {
  cpu_correlation_map_.clear();
  user_correlation_map_.clear();
}

} // namespace KINETO_NAMESPACE
