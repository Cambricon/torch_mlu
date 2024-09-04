#pragma once

#include <cnpapi.h>
#include <string>
#include <chrono>
#include <variant>

#include "TimeGapCleaner.h"

namespace KINETO_NAMESPACE {

struct KernelRecord {
  uint64_t correlation_id;
  uint64_t start;
  uint64_t end;
  uint64_t queued;
  uint64_t device_id;
  std::string name;
  uint64_t queue_id;
  uint64_t kernel_type;
  uint32_t dimx;
  uint32_t dimy;
  uint32_t dimz;
  uint64_t context_id;
  uint32_t tasktopo_id;
  uint64_t tasktopo_node_id;

  KernelRecord(
    uint64_t correlation_id,
    uint64_t start,
    uint64_t end,
    uint64_t queued,
    uint64_t device_id,
    const char* name,
    uint64_t queue_id,
    uint64_t kernel_type,
    uint32_t dimx,
    uint32_t dimy,
    uint32_t dimz,
    uint64_t context_id,
    uint32_t tasktopo_id,
    uint64_t tasktopo_node_id
  ) : correlation_id(correlation_id),
    start(removeTimeGap(start)),
    end(removeTimeGap(end)),
    queued(queued),
    device_id(device_id),
    name(name),
    queue_id(queue_id),
    kernel_type(kernel_type),
    dimx(dimx),
    dimy(dimy),
    dimz(dimz),
    context_id(context_id),
    tasktopo_id(tasktopo_id),
    tasktopo_node_id(tasktopo_node_id)
  {}
};

struct RuntimeRecord {
  cnpapiActivityType type;
  uint64_t correlation_id;
  cnpapi_CallbackId cbid;
  uint64_t start;
  uint64_t end;
  uint32_t process_id;
  uint32_t thread_id;

  RuntimeRecord(
    cnpapiActivityType type,
    uint64_t correlation_id,
    cnpapi_CallbackId cbid,
    uint64_t start,
    uint64_t end,
    uint32_t process_id,
    uint32_t thread_id
  ) : type(type),
    correlation_id(correlation_id),
    cbid(cbid),
    start(removeTimeGap(start)),
    end(removeTimeGap(end)),
    process_id(process_id),
    thread_id(thread_id)
  {}
};

struct OverheadRecord {
  cnpapiActivityOverheadType overhead_type;
  uint64_t start;
  uint64_t end;

  OverheadRecord(
    cnpapiActivityOverheadType overhead_type,
    uint64_t start,
    uint64_t end
  ) : overhead_type(overhead_type),
    start(removeTimeGap(start)),
    end(removeTimeGap(end))
  {}
};

struct MemcpyRecord {
  uint64_t correlation_id;
  uint64_t bytes;
  cnpapiActivityMemcpyType copy_type;
  uint64_t start;
  uint64_t end;
  uint64_t device_id;
  uint64_t queue_id;
  uint64_t context_id;

  MemcpyRecord(
    uint64_t correlation_id,
    uint64_t bytes,
    cnpapiActivityMemcpyType copy_type,
    uint64_t start,
    uint64_t end,
    uint64_t device_id,
    uint64_t queue_id,
    uint64_t context_id
  ) : correlation_id(correlation_id),
    bytes(bytes),
    copy_type(copy_type),
    start(removeTimeGap(start)),
    end(removeTimeGap(end)),
    device_id(device_id),
    queue_id(queue_id),
    context_id(context_id)
  {}
};

struct MemsetRecord {
  uint64_t correlation_id;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint64_t device_id;
  uint64_t queue_id;
  uint64_t context_id;

  MemsetRecord(
    uint64_t correlation_id,
    uint64_t bytes,
    uint64_t start,
    uint64_t end,
    uint64_t device_id,
    uint64_t queue_id,
    uint64_t context_id
  ) : correlation_id(correlation_id),
    bytes(bytes),
    start(removeTimeGap(start)),
    end(removeTimeGap(end)),
    device_id(device_id),
    queue_id(queue_id),
    context_id(context_id)
  {}
};

struct MemcpyP2PRecord {
  uint64_t correlation_id;
  uint64_t bytes;
  cnpapiActivityMemcpyType copy_type;
  uint64_t start;
  uint64_t end;
  uint64_t device_id;
  uint64_t src_device_id;
  uint64_t dst_device_id;
  uint64_t queue_id;
  uint64_t context_id;

  MemcpyP2PRecord(
    uint64_t correlation_id,
    uint64_t bytes,
    cnpapiActivityMemcpyType copy_type,
    uint64_t start,
    uint64_t end,
    uint64_t device_id,
    uint64_t src_device_id,
    uint64_t dst_device_id,
    uint64_t queue_id,
    uint64_t context_id
  ) : correlation_id(correlation_id),
    bytes(bytes),
    copy_type(copy_type),
    start(removeTimeGap(start)),
    end(removeTimeGap(end)),
    device_id(device_id),
    src_device_id(src_device_id),
    dst_device_id(dst_device_id),
    queue_id(queue_id),
    context_id(context_id)
  {}
};

struct ExternalCorrelationRecord {
  cnpapiActivityExternalCorrelationType external_type;
  uint64_t external_id;
  uint64_t correlation_id;

  ExternalCorrelationRecord(
    cnpapiActivityExternalCorrelationType external_type,
    uint64_t external_id,
    uint64_t correlation_id
  ) : external_type(external_type),
    external_id(external_id),
    correlation_id(correlation_id)
  {}
};

using AtomicFlagType = std::variant<cnpapiActivityAtomicRequestFlag,
                                    cnpapiActivityAtomicCompareFlag>;
struct AtomicOpRecord {
  uint64_t correlation_id;
  uint64_t start;
  uint64_t end;
  uint64_t queued;
  uint64_t device_id;
  uint64_t queue_id;
  uint64_t context_id;
  cnpapiActivityAtomicOpType atomic_type;
  AtomicFlagType atomic_flag;
  uint64_t value;

  AtomicOpRecord(
    uint64_t correlation_id,
    uint64_t start,
    uint64_t end,
    uint64_t queued,
    uint64_t device_id,
    uint64_t queue_id,
    uint64_t context_id,
    cnpapiActivityAtomicOpType atomic_type,
    AtomicFlagType atomic_flag,
    uint64_t value
  ) : correlation_id(correlation_id),
    start(removeTimeGap(start)),
    end(removeTimeGap(end)),
    queued(queued),
    device_id(device_id),
    queue_id(queue_id),
    context_id(context_id),
    atomic_type(atomic_type),
    atomic_flag(atomic_flag),
    value(value)
  {}
};

struct CnpapiRecordSet {
  std::vector<KernelRecord> kernel_records;
  std::vector<RuntimeRecord> runtime_records;
  std::vector<OverheadRecord> overhead_records;
  std::vector<MemcpyRecord> memcpy_records;
  std::vector<MemsetRecord> memset_records;
  std::vector<MemcpyP2PRecord> memcpy_p2p_records;
  std::vector<ExternalCorrelationRecord> external_correlation_records;
  std::vector<AtomicOpRecord> atomic_op_records;
};

}  // namespace KINETO_NAMESPACE
