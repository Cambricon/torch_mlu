 /*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CnpapiActivity.h"

#include <algorithm>
#include <utility>

#include "ApiListCommon.h"
#include "MluDeviceProperties.h"
#include "Demangle.h"
#include "output_base.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

template<>
inline const std::string MluActivity<KernelRecord>::name() const {
  return demangle(raw().name);
}

template<>
inline ActivityType MluActivity<KernelRecord>::type() const {
  return ActivityType::MLU_CONCURRENT_KERNEL;
}

template<class T>
inline void MluActivity<T>::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

template<>
inline const std::string MluActivity<KernelRecord>::appendTasktopoExternalOpMetadataJson() const {
  std::pair<int64_t, std::string> id_name;
  if (linked_ == nullptr && raw().tasktopo_node_id > 0) {
    id_name = CnpapiResourceApi::singleton()
        .getExternalIdAndName(raw().tasktopo_node_id);
  }
  if (id_name.second.empty()) {
    return "";
  }
  std::ostringstream op_id_name_meta_data;
  // need 6 spaces for indentation to align with other metadata
  op_id_name_meta_data << ",\n      \"tasktopo_external_id\": "
                    << id_name.first
                    << ",\n      \"tasktopo_external_op\": \""
                    << id_name.second
                    << "\"";
  return op_id_name_meta_data.str();
}

template<>
inline const std::string MluActivity<KernelRecord>::metadataJson() const {
  const KernelRecord& kernel = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "queued": {}, "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "kernel type": "{}",
      "dimx": {}, "dimy": {}, "dimz": {},
      "tasktopo": {},
      "tasktopo_node": {}{}{})JSON",
      kernel.queued, kernel.device_id, kernel.context_id,
      kernel.queue_id, kernel.correlation_id,
      kernelTypeString(kernel.kernel_type),
      kernel.dimx, kernel.dimy, kernel.dimz,
      kernel.tasktopo_id,
      kernel.tasktopo_node_id,
      appendTasktopoExternalOpMetadataJson(),
      appendPmuMetadataJson());
  // clang-format on
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

template<>
inline const std::string MluActivity<AtomicOpRecord>::name() const {
  return demangle(atomicOpName(raw().atomic_type, raw().atomic_flag));
}

template<>
inline ActivityType MluActivity<AtomicOpRecord>::type() const {
  return ActivityType::MLU_CONCURRENT_KERNEL;
}

template<>
inline const std::string MluActivity<AtomicOpRecord>::metadataJson() const {
  const AtomicOpRecord& atomic_op = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "queued": {}, "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "atomic operator type": "{}", "value": {})JSON",
      atomic_op.queued, atomic_op.device_id, atomic_op.context_id,
      atomic_op.queue_id, atomic_op.correlation_id,
      atomicOpTypeString(atomic_op.atomic_type), atomic_op.value);
  // clang-format on
}

inline std::string memcpyName(uint64_t kind) {
  return fmt::format(
      "Memcpy {}",
      memcpyKindString((cnpapiActivityMemcpyType)kind));
}

template<>
inline ActivityType MluActivity<MemcpyRecord>::type() const {
  return ActivityType::MLU_MEMCPY;
}

template<>
inline const std::string MluActivity<MemcpyRecord>::name() const {
  return memcpyName(raw().copy_type);
}

inline std::string bandwidth(uint64_t bytes, uint64_t duration) {
  return duration == 0 ? "\"N/A\"" : fmt::format("{}", bytes * 1.0 / duration);
}

template<>
inline const std::string MluActivity<MemcpyRecord>::metadataJson() const {
  const MemcpyRecord& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memcpy.device_id, memcpy.context_id,
      memcpy.queue_id, memcpy.correlation_id,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on
}


template<>
inline ActivityType MluActivity<MemcpyP2PRecord>::type() const {
  return ActivityType::MLU_MEMCPY;
}

template<>
inline const std::string MluActivity<MemcpyP2PRecord>::name() const {
  return memcpyName(raw().copy_type);
}

template<>
inline const std::string MluActivity<MemcpyP2PRecord>::metadataJson() const {
  const MemcpyP2PRecord& memcpy = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "fromDevice": {}, "inDevice": {}, "toDevice": {},
      "Context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memcpy.src_device_id, memcpy.device_id, memcpy.dst_device_id,
      memcpy.context_id,
      memcpy.queue_id, memcpy.correlation_id,
      memcpy.bytes, bandwidth(memcpy.bytes, memcpy.end - memcpy.start));
  // clang-format on
}

template<>
inline const std::string MluActivity<MemsetRecord>::name() const {
  return fmt::format("Memset");
}

template<>
inline ActivityType MluActivity<MemsetRecord>::type() const {
  return ActivityType::MLU_MEMSET;
}

template<>
inline const std::string MluActivity<MemsetRecord>::metadataJson() const {
  const MemsetRecord& memset = raw();
  // clang-format off
  return fmt::format(R"JSON(
      "device": {}, "context": {},
      "stream": {}, "correlation": {},
      "bytes": {}, "memory bandwidth (GB/s)": {})JSON",
      memset.device_id, memset.context_id,
      memset.queue_id, memset.correlation_id,
      memset.bytes, bandwidth(memset.bytes, memset.end - memset.start));
  // clang-format on
}

inline void RuntimeActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline void OverheadActivity::log(ActivityLogger& logger) const {
  logger.handleActivity(*this);
}

inline bool OverheadActivity::flowStart() const {
  return false;
}

inline const std::string OverheadActivity::metadataJson() const {
  return "";
}

inline bool RuntimeActivity::flowStart() const {
  const auto& launch_apis = ApiList::getInstance().getCndrvLaunchCbidList();
  if (std::find(launch_apis.begin(), launch_apis.end(), activity_.cbid) != launch_apis.end()) {
    return true;
  } else {
    return false;
  }
}

inline const std::string RuntimeActivity::metadataJson() const {
  return fmt::format(R"JSON("cbid": {}, "correlation": {})JSON",
      activity_.cbid, activity_.correlation_id);
}

template<class T>
inline const std::string MluActivity<T>::metadataJson() const {
  return "";
}

} // namespace KINETO_NAMESPACE
