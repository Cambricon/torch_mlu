#include "CnperfActivity.h"

#include <algorithm>
#include <variant>

#include "MluDeviceProperties.h"
#include "Logger.h"

namespace KINETO_NAMESPACE {
namespace {

const std::string getPmuMetadataJson(const std::vector<CnperfPmuData>* pmu_data) {
  std::ostringstream fmt_pmu_data;
  // need 6 spaces for indentation to align with other metadata
  fmt_pmu_data << "{";
  bool is_first = true;
  for (const auto& pmu : *pmu_data) {
    if (!is_first) {
      fmt_pmu_data << ",";
    }
    fmt_pmu_data << "\"" << pmu.counter_name << "\": ";
    std::visit([&](const auto &v) { fmt_pmu_data << v; }, pmu.value);
    is_first = false;
  }
  fmt_pmu_data << "}";
  return fmt_pmu_data.str();
}

void addDeviceTaskMetaData(GenericTraceActivity& generic_activity,
                       const DeviceTaskRecord* device_task,
                       std::vector<CnperfPmuData>* pmu_data) {
  generic_activity.addMetadata("device", device_task->device_id);
  generic_activity.addMetadata("context", device_task->context_id);
  generic_activity.addMetadata("stream", device_task->queue_id);
  generic_activity.addMetadata("correlation", device_task->correlation_id);
  generic_activity.addMetadata("tasktopo", device_task->tasktopo_id);
  generic_activity.addMetadata("tasktopo node", device_task->tasktopo_node_id);
  generic_activity.addMetadata("is_async", device_task->is_async);
  if (!device_task->tasktopo_external_op_name.empty()) {
    generic_activity.addMetadataQuoted("tasktopo external op",
                                       device_task->tasktopo_external_op_name);
    generic_activity.addMetadata("tasktopo external id",
                                 device_task->tasktopo_external_op_id);
  }
  if (!device_task->extra.empty()) {
    generic_activity.addMetadata("extra", device_task->extra);
  }
  if (pmu_data && !pmu_data->empty()) {
    generic_activity.addMetadata("pmus", getPmuMetadataJson(pmu_data));
  }
}

std::string getCommType(cnperfDataCommTaskType_t type) {
  if (type == CNPERF_DATA_COMM_TASK_TYPE_P2P) {
    return "P2P";
  } else if (type == CNPERF_DATA_COMM_TASK_TYPE_COLLECTIVE) {
    return "COLLETIVE";
  } else {
    return "UNKNOWN";
  }
}

}  // namespace


GenericTraceActivity createRuntimeActivity(
    const RuntimeRecord* activity,
    const ITraceActivity* linked,
    int32_t threadId,
    bool flow_start) {
  GenericTraceActivity runtime_activity;
  runtime_activity.activityName = activity->name;
  runtime_activity.startTime = activity->start;
  runtime_activity.endTime = activity->end;
  runtime_activity.id = activity->correlation_id;
  runtime_activity.device = processId();
  runtime_activity.resource = threadId;
  runtime_activity.activityType = ActivityType::MLU_RUNTIME;
  runtime_activity.linked = linked;
  runtime_activity.flow.id = activity->correlation_id;
  runtime_activity.flow.type = kLinkAsyncCpuMlu;
  // flow.start = true if runtime api can launch device_task
  runtime_activity.flow.start = flow_start;
  runtime_activity.addMetadata("correlation", activity->correlation_id);
  if (!activity->extra.empty()) {
    runtime_activity.addMetadata("extra", activity->extra);
  }

  return runtime_activity;
}

GenericTraceActivity createMluActivity(
    const DeviceTaskRecord* activity,
    const ITraceActivity* linked,
    std::vector<CnperfPmuData>* pmu_data) {
  GenericTraceActivity mlu_activity;
  mlu_activity.activityName = activity->name;
  mlu_activity.startTime = activity->start;
  mlu_activity.endTime = activity->end;
  mlu_activity.id = activity->correlation_id;
  mlu_activity.device = activity->device_id;
  mlu_activity.resource = activity->queue_id;
  mlu_activity.flow.id = activity->correlation_id;
  mlu_activity.flow.type = kLinkAsyncCpuMlu;
  mlu_activity.flow.start = false;
  mlu_activity.linked = linked;
  addDeviceTaskMetaData(mlu_activity, activity, pmu_data);
  switch (activity->type)
  {
  case CNPERF_DATA_DEVICE_TASK_ATOMIC:
  case CNPERF_DATA_DEVICE_TASK_KERNEL:
  case CNPERF_DATA_DEVICE_TASK_HOST_FUNC:
    mlu_activity.activityType = ActivityType::MLU_CONCURRENT_KERNEL;
    break;
  case CNPERF_DATA_DEVICE_TASK_MEMCPY:
    mlu_activity.activityType = ActivityType::MLU_MEMCPY;
    break;
  case CNPERF_DATA_DEVICE_TASK_MEMORY:
    // skip for now.
    break;
  case CNPERF_DATA_DEVICE_TASK_MEMSET:
    mlu_activity.activityType = ActivityType::MLU_MEMSET;
    break;
  case CNPERF_DATA_DEVICE_TASK_NOTIFIER:
    mlu_activity.activityType = ActivityType::MLU_NOTIFIER;
    break;
  default:
    break;
  }

  return mlu_activity;
}

GenericTraceActivity createCommunicationActivity(
    const CommunicationRecord* activity) {
  GenericTraceActivity comm_activity;
  comm_activity.activityName = activity->name;
  comm_activity.startTime = activity->start;
  comm_activity.endTime = activity->end;
  comm_activity.id = activity->external_correlation_id;
  comm_activity.device = activity->rank;
  comm_activity.resource = activity->thread_id;
  comm_activity.activityType = ActivityType::MLU_USER_ANNOTATION;
  comm_activity.linked = nullptr;
  comm_activity.flow.id = activity->external_correlation_id;
  comm_activity.flow.type = kLinkAsyncCpuMlu;
  comm_activity.flow.start = false;
  comm_activity.addMetadata("bytes", activity->bytes);
  comm_activity.addMetadata("rank", activity->rank);
  comm_activity.addMetadata("clique id", activity->clique_id);
  comm_activity.addMetadataQuoted("type", getCommType(activity->type));
  comm_activity.addMetadataQuoted("op name", activity->op_name);
  return comm_activity;
}

} // namespace KINETO_NAMESPACE
