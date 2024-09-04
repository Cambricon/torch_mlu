/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MluDeviceProperties.h"

#include <fmt/format.h>
#include <vector>

#include <cnrt.h>

#include "Logger.h"

namespace libkineto_mlu {

static const std::vector<cnrtDeviceProp_t> createDeviceProps() {
  std::vector<cnrtDeviceProp_t> props;
  unsigned int device_count;
  cnrtRet_t error_id = cnrtGetDeviceCount(&device_count);
  // Return empty vector if error.
  if (error_id != cnrtSuccess) {
    LOG(ERROR) << "cnrtGetDeviceCount failed with code " << error_id;
    return {};
  }
  LOG(INFO) << "Device count is " << device_count;
  for (size_t i = 0; i < device_count; ++i) {
    cnrtDeviceProp_t prop;
    error_id = cnrtGetDeviceProperties(&prop, i);
    // Return empty vector if any device property fail to get.
    if (error_id != cnrtSuccess) {
      LOG(ERROR) << "cnrtGetDeviceProperties failed with " << error_id;
      return {};
    }
    props.push_back(prop);
    LOGGER_OBSERVER_ADD_DEVICE(i);
  }
  return props;
}

static const std::vector<cnrtDeviceProp_t>& deviceProps() {
  static const std::vector<cnrtDeviceProp_t> props = createDeviceProps();
  return props;
}

static const std::string createDevicePropertiesJson(
    size_t id, const cnrtDeviceProp_t& props) {
  return fmt::format(R"JSON(
    {{
      "id": {}, "name": "{}", "totalMem(MiB)": {},
      "computeMajor": {}, "computeMinor": {},
      "FP16ComputingSupported": {}, "INT4ComputingSupported": {},
      "INT8ComputingSupported": {}, "BF16ComputingSupported": {},
      "TF32ComputingSupported": {}, "clusterCount": {},
      "maxClusterCountPerUnionTask": {}, "McorePerCluster": {},
      "maxQuadrantCount": {}, "maxUnionTypePerQuadrant": {},
      "maxClusterPerUnionLimitTask": {}
    }})JSON",
      id, props.name, props.totalMem,
      props.major, props.minor,
      props.FP16ComputingSupported, props.INT4ComputingSupported,
      props.INT8ComputingSupported, props.BF16ComputingSupported,
      props.TF32ComputingSupported, props.clusterCount,
      props.maxClusterCountPerUnionTask, props.McorePerCluster,
      props.maxQuadrantCount, props.maxUnionTypePerQuadrant,
      props.maxClusterPerUnionLimitTask);
}

static const std::string createDevicePropertiesJson() {
  std::vector<std::string> jsonProps;
  const auto& props = deviceProps();
  for (size_t i = 0; i < props.size(); i++) {
    jsonProps.push_back(createDevicePropertiesJson(i, props[i]));
  }
  return fmt::format("{}", fmt::join(jsonProps, ","));
}

const std::string& devicePropertiesJson() {
  static std::string devicePropsJson = createDevicePropertiesJson();
  return devicePropsJson;
}

} // namespace libkineto_mlu
