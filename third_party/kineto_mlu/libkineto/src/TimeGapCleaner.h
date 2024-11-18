# pragma once

#include <chrono>

namespace KINETO_NAMESPACE {

class TimeGap {
  public:
    static TimeGap& instance();

    uint64_t get() {
        return time_gap_;
    }

    ~TimeGap() = default;

  private:
    TimeGap();

    uint64_t time_gap_;
};

inline uint64_t transCPUTimeToCnperfTime(uint64_t cpu_ts, uint64_t start_ts) {
  return cpu_ts - TimeGap::instance().get() - start_ts;
}

inline uint64_t transCnperfTimeToCPUTime(uint64_t cnperf_ts, uint64_t start_ts) {
  return cnperf_ts + TimeGap::instance().get() + start_ts;

}

} // namespace KINETO_NAMESPACE

