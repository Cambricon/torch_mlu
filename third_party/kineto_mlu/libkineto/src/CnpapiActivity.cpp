/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CnpapiActivity.h"

#include <algorithm>

#include "CnpapiActivityApi.h"
#include "MluDeviceProperties.h"
#include "Demangle.h"

namespace libkineto_mlu {

namespace {

std::string memcpyName(uint64_t kind) {
  return fmt::format(
      "Memcpy {}",
      memcpyKindString((cnpapiActivityMemcpyType)kind));
}

std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

void addKernelMetaData(GenericTraceActivity& generic_activity,
                       const KernelRecord* kernel) {
  generic_activity.addMetadata("queued",
      kernel->queued ? kernel->queued : 0);
  generic_activity.addMetadata("device", kernel->device_id);
  generic_activity.addMetadata("context", kernel->context_id);
  generic_activity.addMetadata("stream", kernel->queue_id);
  generic_activity.addMetadata("correlation", kernel->correlation_id);
  generic_activity.addMetadataQuoted("kernel type", kernelTypeString(kernel->kernel_type));
  generic_activity.addMetadataQuoted("dim",
      fmt::format("[{},{},{}]", kernel->dimx, kernel->dimy, kernel->dimz));
  generic_activity.addMetadata("tasktopo", kernel->tasktopo_id);
  generic_activity.addMetadata("tasktopo node", kernel->tasktopo_node_id);
}

void addMemcpyMetaData(GenericTraceActivity& generic_activity,
                       const MemcpyRecord* memcpy) {
  generic_activity.addMetadata("device", memcpy->device_id);
  generic_activity.addMetadata("context", memcpy->context_id);
  generic_activity.addMetadata("stream", memcpy->queue_id);
  generic_activity.addMetadata("correlation", memcpy->correlation_id);
  generic_activity.addMetadata("bytes", memcpy->bytes);
  generic_activity.addMetadata("memory bandwidth (GB/s)",
                       bandwidth(memcpy->bytes, memcpy->end - memcpy->start));
}

void addMemcpyP2PMetaData(GenericTraceActivity& generic_activity,
                          const MemcpyP2PRecord* memcpy) {
  generic_activity.addMetadata("fromDevice", memcpy->src_device_id);
  generic_activity.addMetadata("inDevice", memcpy->device_id);
  generic_activity.addMetadata("toDevice", memcpy->dst_device_id);
  generic_activity.addMetadata("context", memcpy->context_id);
  generic_activity.addMetadata("stream", memcpy->queue_id);
  generic_activity.addMetadata("correlation", memcpy->correlation_id);
  generic_activity.addMetadata("bytes", memcpy->bytes);
  generic_activity.addMetadata("memory bandwidth (GB/s)",
                       bandwidth(memcpy->bytes, memcpy->end - memcpy->start));
}

void addMemsetMetaData(GenericTraceActivity& generic_activity,
                       const MemsetRecord* memset) {
  generic_activity.addMetadata("device", memset->device_id);
  generic_activity.addMetadata("context", memset->context_id);
  generic_activity.addMetadata("stream", memset->queue_id);
  generic_activity.addMetadata("correlation", memset->correlation_id);
  generic_activity.addMetadata("bytes", memset->bytes);
  generic_activity.addMetadata("memory bandwidth (GB/s)",
                       bandwidth(memset->bytes, memset->end - memset->start));
}

std::string atomicOpName(cnpapiActivityAtomicOpType type,
                         AtomicFlagType flag) {
  if (type == CNPAPI_ACTIVITY_ATOMIC_OP_REQUEST) {
    return fmt::format("Atomic Operation[{}]",
      atomicRequestFlagString(std::get<cnpapiActivityAtomicRequestFlag>(flag)));
  } else if (type == CNPAPI_ACTIVITY_ATOMIC_OP_COMPARE) {
    return fmt::format("Atomic Operation[{}]",
      atomicCompareFlagString(std::get<cnpapiActivityAtomicCompareFlag>(flag)));
  } else {
    return "Atomic Operation[unknown]";
  }
}

void addAtomicOpMetaData(GenericTraceActivity& generic_activity,
                         const AtomicOpRecord* atomic_op) {
  generic_activity.addMetadata("queued",
      atomic_op->queued ? atomic_op->queued : 0);
  generic_activity.addMetadata("device", atomic_op->device_id);
  generic_activity.addMetadata("context", atomic_op->context_id);
  generic_activity.addMetadata("stream", atomic_op->queue_id);
  generic_activity.addMetadata("correlation", atomic_op->correlation_id);
  generic_activity.addMetadataQuoted("atomic operator type",
      atomicOpTypeString(atomic_op->atomic_type));
  generic_activity.addMetadata("value", atomic_op->value);
}

}  // namespace


GenericTraceActivity createRuntimeActivity(
    const RuntimeRecord* activity,
      const ITraceActivity* linked,
      int32_t threadId) {
  GenericTraceActivity runtime_activity;
  runtime_activity.activityName = runtimeCbidName(activity->type,
                                                  activity->cbid);
  runtime_activity.startTime = activity->start;
  runtime_activity.endTime = activity->end;
  runtime_activity.id = activity->correlation_id;
  runtime_activity.device = processId();
  runtime_activity.resource = threadId;
  runtime_activity.activityType = ActivityType::PRIVATEUSE1_RUNTIME;
  runtime_activity.linked = linked;
  runtime_activity.flow.id = activity->correlation_id;
  runtime_activity.flow.type = kLinkAsyncCpuGpu;
  // flow.start = true if runtime api can launch kernel
  runtime_activity.flow.start =
    [&]() {
      const auto& launch_apis = CnpapiActivityApi::singleton().getCndrvLaunchCbidList();
      if (std::find(launch_apis.begin(), launch_apis.end(), activity->cbid) != launch_apis.end()) {
        return true;
      } else {
        return false;
      }
    } ();
    (activity->cbid == CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel ||
     activity->cbid == CNPAPI_CNDRV_TRACE_CBID_cnQueueAtomicOperation ||
     activity->cbid == CNPAPI_CNDRV_TRACE_CBID_cnTaskTopoEntityInvoke ||
    (activity->cbid >= CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync &&
     activity->cbid <= CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32Async) ||
    (activity->cbid >= CNPAPI_CNDRV_TRACE_CBID_cnMemcpy &&
     activity->cbid <= CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32) ||
    (activity->cbid >= CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync_V2 &&
     activity->cbid <= CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoHAsync_V2));
  runtime_activity.addMetadata("cbid", activity->cbid);
  runtime_activity.addMetadata("correlation", activity->correlation_id);

  return runtime_activity;
}

GenericTraceActivity createOverheadActivity(
      const OverheadRecord* activity,
      const ITraceActivity* linked,
      int32_t threadId) {
  GenericTraceActivity overhead_activity;
  overhead_activity.activityName = overheadKindString(activity->overhead_type);
  overhead_activity.startTime = activity->start;
  overhead_activity.endTime = activity->end;
  overhead_activity.id = 0;
  overhead_activity.device = -1;
  overhead_activity.resource = threadId;
  overhead_activity.activityType = ActivityType::OVERHEAD;
  overhead_activity.linked = linked;
  overhead_activity.flow.id = 0;
  overhead_activity.flow.type = kLinkAsyncCpuGpu;
  overhead_activity.flow.start = false;

  return overhead_activity;
}

template<class T>
GenericTraceActivity createMluActivity(
      const T* activity, const ITraceActivity* linked) {
  GenericTraceActivity mlu_activity;
  mlu_activity.startTime = activity->start;
  mlu_activity.endTime = activity->end;
  mlu_activity.id = activity->correlation_id;
  mlu_activity.device = activity->device_id;
  mlu_activity.resource = activity->queue_id;
  mlu_activity.flow.id = activity->correlation_id;
  mlu_activity.flow.type = kLinkAsyncCpuGpu;
  mlu_activity.flow.start = false;
  mlu_activity.linked = linked;

  if constexpr (std::is_same_v<T, KernelRecord>) {
    mlu_activity.activityName = demangle(activity->name);
    mlu_activity.activityType = ActivityType::CONCURRENT_KERNEL;
    addKernelMetaData(mlu_activity, activity);
  } else if constexpr (std::is_same_v<T, MemcpyRecord>) {
    mlu_activity.activityName = memcpyName(activity->copy_type);
    mlu_activity.activityType = ActivityType::GPU_MEMCPY;
    addMemcpyMetaData(mlu_activity, activity);
  } else if constexpr (std::is_same_v<T, MemcpyP2PRecord>) {
    mlu_activity.activityName = memcpyName(activity->copy_type);
    mlu_activity.activityType = ActivityType::GPU_MEMCPY;
    addMemcpyP2PMetaData(mlu_activity, activity);
  } else if constexpr (std::is_same_v<T, MemsetRecord>) {
    mlu_activity.activityName = "Memset";
    mlu_activity.activityType = ActivityType::GPU_MEMSET;
    addMemsetMetaData(mlu_activity, activity);
  } else if constexpr (std::is_same_v<T, AtomicOpRecord>) {
    mlu_activity.activityName = atomicOpName(activity->atomic_type, activity->atomic_flag);
    mlu_activity.activityType = ActivityType::CONCURRENT_KERNEL;
    addAtomicOpMetaData(mlu_activity, activity);
  }

  return mlu_activity;
}

template GenericTraceActivity createMluActivity<KernelRecord>(
  const KernelRecord* activity, const ITraceActivity* linked);
template GenericTraceActivity createMluActivity<MemcpyRecord>(
  const MemcpyRecord* activity, const ITraceActivity* linked);
template GenericTraceActivity createMluActivity<MemcpyP2PRecord>(
  const MemcpyP2PRecord* activity, const ITraceActivity* linked);
template GenericTraceActivity createMluActivity<MemsetRecord>(
  const MemsetRecord* activity, const ITraceActivity* linked);
template GenericTraceActivity createMluActivity<AtomicOpRecord>(
  const AtomicOpRecord* activity, const ITraceActivity* linked);

} // namespace libkineto_mlu
