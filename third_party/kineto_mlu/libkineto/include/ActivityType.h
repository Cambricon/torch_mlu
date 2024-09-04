// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <array>
#include <string>

namespace libkineto {

// Note : All activity types are not enabled by default. Please add them
// at correct position in the enum
enum class ActivityType {
    // Activity types enabled by default
    CPU_OP = 0, // cpu side ops
    USER_ANNOTATION,
    GPU_USER_ANNOTATION,
    GPU_MEMCPY,
    GPU_MEMSET,
    CONCURRENT_KERNEL, // on-device kernels
    EXTERNAL_CORRELATION,
    CUDA_RUNTIME, // host side cuda runtime events
    CUDA_DRIVER, // host side cuda driver events
    CPU_INSTANT_EVENT, // host side point-like events
    PYTHON_FUNCTION,
    OVERHEAD, // CUPTI induced overhead events sampled from its overhead API.
    MTIA_RUNTIME, // host side MTIA runtime events
    MTIA_CCP_EVENTS, // MTIA ondevice CCP events

    // Optional Activity types
    CUDA_SYNC, // synchronization events between runtime and kernels
    GLOW_RUNTIME, // host side glow runtime events
    CUDA_PROFILER_RANGE, // CUPTI Profiler range for performance metrics
    HPU_OP, // HPU host side runtime event
    XPU_RUNTIME, // host side xpu runtime events
    COLLECTIVE_COMM, // collective communication

    // Activity types for MLU
    MLU_USER_ANNOTATION,
    MLU_MEMCPY,
    MLU_MEMSET,
    MLU_CONCURRENT_KERNEL, // on-device kernels
    MLU_RUNTIME,
    MLU_PROFILER_RANGE,

    ENUM_COUNT, // This is to add buffer and not used for any profiling logic. Add your new type before it.
    OPTIONAL_ACTIVITY_TYPE_START = CUDA_SYNC,
};

const char* toString(ActivityType t);
ActivityType toActivityType(const std::string& str);

// Return an array of all activity types except COUNT
constexpr int activityTypeCount = (int)ActivityType::ENUM_COUNT;
constexpr int defaultActivityTypeCount = (int)ActivityType::OPTIONAL_ACTIVITY_TYPE_START;
const std::array<ActivityType, activityTypeCount> activityTypes();
const std::array<ActivityType, defaultActivityTypeCount> defaultActivityTypes();

} // namespace libkineto
