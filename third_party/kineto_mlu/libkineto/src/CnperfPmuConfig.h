#pragma once

#include <Config.h>

#include <chrono>
#include <string>
#include <vector>

namespace KINETO_NAMESPACE {

using namespace libkineto;

class CnperfPmuConfig : public AbstractConfig {
 public:
  CnperfPmuConfig();
  ~CnperfPmuConfig() = default;

  bool valid(bool custom_config=false);

  std::string getCounterString();

  bool handleOption(const std::string& name, std::string& val) override;

  void validate(
      const std::chrono::time_point<std::chrono::system_clock>&
      fallbackProfileStartTime) override {}

 protected:
  AbstractConfig* cloneDerived(AbstractConfig& parent) const override {}

 private:
  bool envEnabled_{false};
  std::string configuredPmuCounters_;
  // Collect pmu metrics per kernel, only support true and need set explicit.
  bool pmuPerKernel_{false};
};

} // namespace KINETO_NAMESPACE