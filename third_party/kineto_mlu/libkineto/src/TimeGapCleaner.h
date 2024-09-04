# pragma once

#include <chrono>

namespace KINETO_NAMESPACE {

class TimeGap {
  public:
    static TimeGap& instance();

    int64_t get() {
        return time_gap_;
    }

    ~TimeGap() = default;

  private:
    TimeGap();

    int64_t time_gap_;
};

inline int64_t removeTimeGap(int64_t timestamp) {
  return timestamp + TimeGap::instance().get();
}

} // namespace KINETO_NAMESPACE

