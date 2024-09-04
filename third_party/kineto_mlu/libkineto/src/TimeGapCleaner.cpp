#include "TimeGapCleaner.h"

#include <cnpapi.h>

namespace libkineto_mlu {

TimeGap& TimeGap::instance() {
    static TimeGap instance;
    return instance;
}

TimeGap::TimeGap() {
    // Timestamp of cnpapi is inconsistent with cpu's.
    // Get the timestamps of cpu and mlu at the same time respectively
    // and apply the time_gap to every records of mlu.
    time_gap_ = [] {
        // Add warmup to reduce overhead of calling these 2 apis
        // to make time_gap more accurate.
        for (int i = 0; i < 5; ++i) {
            cnpapiGetTimestamp();
            std::chrono::high_resolution_clock::now();
        }
        uint64_t cnpapi_time = cnpapiGetTimestamp();
        auto time_cpu = std::chrono::high_resolution_clock::now();
        return (std::chrono::duration_cast<std::chrono::nanoseconds>(
            time_cpu.time_since_epoch()).count() - cnpapi_time);
    }();
}

} // namespace libkineto_mlu
