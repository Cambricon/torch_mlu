#pragma once

#include <cnperf_api.h>
#include <fmt/format.h>

#include <chrono>
#include <sstream>

#include "ITraceActivity.h"
#include "GenericTraceActivity.h"
#include "ThreadUtil.h"
#include "CnperfRecord.h"
#include "CnperfPmuApi.h"
#include "time_since_epoch.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

GenericTraceActivity createRuntimeActivity(
    const RuntimeRecord* activity,
    const ITraceActivity* linked,
    int32_t threadId,
    bool flow_start);

GenericTraceActivity createMluActivity(
    const DeviceTaskRecord* activity,
    const ITraceActivity* linked,
    std::vector<CnperfPmuData>* pmu_data);

GenericTraceActivity createCommunicationActivity(
    const CommunicationRecord* activity);

} // namespace KINETO_NAMESPACE
