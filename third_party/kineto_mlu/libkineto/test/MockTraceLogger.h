#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Config.h>
#include <GenericTraceActivity.h>
#include <output_base.h>

namespace libkineto_mlu {
using namespace libkineto;
using KINETO_NAMESPACE::ActivityBuffers;

class MockTraceLogger : public ActivityLogger {
 public:
  MockTraceLogger() {
    activities_.reserve(100000);
  }
  void handleDeviceInfo(
      const DeviceInfo &info,
      uint64_t time) {};

  void handleResourceInfo(const ResourceInfo& info, int64_t time) {};

  void handleOverheadInfo(const OverheadInfo& info, int64_t time) {};

  void handleTraceSpan(const TraceSpan& span) {};

  void handleGenericActivity(
      const libkineto::GenericTraceActivity& activity) {};

  void handleTraceStart(
      const std::unordered_map<std::string, std::string>& metadata) {};

  void handleTraceStart() {
    handleTraceStart(std::unordered_map<std::string, std::string>());
  }

  void finalizeTrace(
      const Config& config,
      std::unique_ptr<ActivityBuffers> buffers,
      int64_t endTime,
      std::unordered_map<std::string, std::vector<std::string>>& metadata) {};

  // Just add the pointer to the list - ownership of the underlying
  // objects must be transferred in ActivityBuffers via finalizeTrace
  void handleActivity(const ITraceActivity& activity) override {
    activities_.push_back(&activity);
  }

  std::vector<const ITraceActivity*> getActivities() {
    return activities_;
  }

 private:

  std::unique_ptr<Config> config_;
  // Optimization: Remove unique_ptr by keeping separate vector per type
  std::vector<const ITraceActivity*> activities_;
};

} // namespace libkineto_mlu
