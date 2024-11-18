#pragma once

#include <cnperf_api.h>

#include <chrono>
#include <string>
#include <vector>

namespace KINETO_NAMESPACE {

struct DeviceTaskRecord {
  cnperfDataDeviceTaskType_t type;
  uint64_t process_id;
  uint64_t correlation_id;
  uint64_t start;
  uint64_t end;
  uint64_t device_id;
  uint64_t context_id;
  uint64_t queue_id;
  uint32_t tasktopo_id;
  uint64_t tasktopo_node_id;
  uint64_t is_async;
  std::string name;
  std::string extra;
  uint64_t tasktopo_external_op_id;
  std::string tasktopo_external_op_name;

  DeviceTaskRecord(
    cnperfDataDeviceTaskType_t type,
    uint64_t process_id,
    uint64_t correlation_id,
    uint64_t start,
    uint64_t end,
    uint64_t device_id,
    uint64_t context_id,
    uint64_t queue_id,
    uint32_t tasktopo_id,
    uint64_t tasktopo_node_id,
    uint64_t is_async,
    const char* name,
    const char* extra,
    uint64_t tasktopo_external_op_id,
    std::string tasktopo_external_op_name
  ) : type(type),
    process_id(process_id),
    correlation_id(correlation_id),
    start(start),
    end(end),
    device_id(device_id),
    context_id(context_id),
    queue_id(queue_id),
    tasktopo_id(tasktopo_id),
    tasktopo_node_id(tasktopo_node_id),
    is_async(is_async),
    name(name),
    extra(extra),
    tasktopo_external_op_id(tasktopo_external_op_id),
    tasktopo_external_op_name(std::move(tasktopo_external_op_name))
  {}
};

struct RuntimeRecord {
  uint64_t correlation_id;
  uint64_t start;
  uint64_t end;
  uint32_t thread_id;
  std::string name;
  std::string extra;

  RuntimeRecord(
    uint64_t correlation_id,
    uint64_t start,
    uint64_t end,
    uint32_t thread_id,
    const char* name,
    const char* extra
  ) : correlation_id(correlation_id),
    start(start),
    end(end),
    thread_id(thread_id),
    name(name),
    extra(extra)
  {}
};

struct CommunicationRecord {
  int32_t thread_id;
  uint64_t external_correlation_id;
  std::string op_name;
  cnperfDataCommTaskType_t type;
  int32_t rank;
  uint64_t clique_id;
  uint64_t start;
  uint64_t end;
  uint64_t bytes;
  std::string name;

  CommunicationRecord() {}
  CommunicationRecord(
    int32_t thread_id,
    uint64_t external_correlation_id,
    const char* op_name,
    cnperfDataCommTaskType_t type,
    int32_t rank,
    uint64_t clique_id,
    uint64_t start,
    uint64_t end,
    uint64_t bytes,
    const char* name
  ) : thread_id(thread_id),
    external_correlation_id(external_correlation_id),
    op_name(op_name),
    type(type),
    rank(rank),
    clique_id(clique_id),
    start(start),
    end(end),
    bytes(bytes),
    name(name)
  {}
};

struct CnperfRecordSet {
  std::vector<DeviceTaskRecord> device_task_records;
  std::vector<RuntimeRecord> runtime_records;
  std::vector<CommunicationRecord> comm_records;
};

}  // namespace KINETO_NAMESPACE
