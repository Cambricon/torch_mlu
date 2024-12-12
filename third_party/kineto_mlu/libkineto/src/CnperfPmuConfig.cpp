#include "CnperfPmuConfig.h"

#include <algorithm>

#include "Logger.h"


namespace {

const std::vector<std::string> DEFAULT_COUNTER_NAMES = {
    "llc__tagram_hit",
    "llc__tagram_miss",
    "llc__eviction",
    "tp_core__lt_cycles",
    "tp_core__csimd_post_cycles",
    "tp_core__vfu_computing_cycles",
    "tp_core__alu_cycles",
    "tp_core__mv_inst_cycles",
    "tp_core__read_bytes",
    "tp_core__write_bytes",
    "tp_cluster__read_bytes",
    "tp_cluster__write_bytes",
    "tp_core__dram_read_cycles",
    "tp_core__dram_write_cycles",
    "tp_core__inst_cache_miss",
    "tp_memcore__dram_read_cycles",
    "tp_memcore__dram_write_cycles"
};

// concat string list using seperator ',': "xxx,xxx,xxx..."
std::string concatPmus(const std::vector<std::string>& pmu_names) {
  std::string result;
  for (auto pmu_name : pmu_names) {
    result.append(pmu_name).append(1, ',');
  }
  result.pop_back();
  return result;
}

static std::string DEFAULT_PMU_COUNTERS = concatPmus(DEFAULT_COUNTER_NAMES);

}  // namespace

namespace KINETO_NAMESPACE {

constexpr char kPmuMetricsKey[] = "CUPTI_PROFILER_METRICS";
constexpr char kPmuPerKernelKey[] = "CUPTI_PROFILER_ENABLE_PER_KERNEL";

CnperfPmuConfig::CnperfPmuConfig() {
  envEnabled_ = [] {
    bool result = false;
    const auto env = std::getenv("ENABLE_CATCHING_PMU_DATA");
    if (env) {
      std::string env_str = env;
      std::transform(env_str.begin(), env_str.end(), env_str.begin(),
                    [](unsigned char c) { return std::tolower(c); });
      if (env_str == "true" || env_str == "1" || env_str == "on") {
        result = true;
      }
    }
    return result;
  }();
}

bool CnperfPmuConfig::handleOption(const std::string& name, std::string& val) {
  LOG(INFO) << " handling : " << name << " = " << val;
  if (!name.compare(kPmuMetricsKey)) {
    configuredPmuCounters_ = concatPmus(splitAndTrim(val, ','));
  } else if (!name.compare(kPmuPerKernelKey)) {
    pmuPerKernel_ = toBool(val);
    if (!pmuPerKernel_) {
      LOG(ERROR) << "MLU not support collect pmu data with custom range, "
                 << "please set profiler_measure_per_kernel=True in experimental_config"
                 << ". Otherwise, the host events will not be collected.";
    }
  }
  return true;
}

bool CnperfPmuConfig::valid(bool custom_config) {
  if (custom_config) {
    return !configuredPmuCounters_.empty();
  }
  return envEnabled_ || !configuredPmuCounters_.empty();
}

std::string CnperfPmuConfig::getCounterString() {
  if (!configuredPmuCounters_.empty()) {
    return configuredPmuCounters_;
  }
  return DEFAULT_PMU_COUNTERS;
}

} // namespace KINETO_NAMESPACE
