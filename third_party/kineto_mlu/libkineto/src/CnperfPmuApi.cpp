#include "CnperfPmuApi.h"

#include <cnrt.h>
#include <algorithm>
#include <cstring>

namespace KINETO_NAMESPACE {

// define default pmu events to be catched
static const std::vector<std::string> SELECTED_PMU_EVENTS_LIST = {
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
  "tp_memcore__dram_write_cycles",
};
// concat string list using seperator ',': "xxx,xxx,xxx..."
static const char* SELECTED_PMU_EVENTS = [](){
  static std::string result;
  for (auto event : SELECTED_PMU_EVENTS_LIST) {
    result.append(event).append(1, ',');
  }
  result.pop_back();
  return result.c_str();
}();

CnperfPmuApi& CnperfPmuApi::singleton() {
  static CnperfPmuApi singleton_;
  return singleton_;
}

void CnperfPmuApi::updateConfig(cnperfConfig_t config) {
  if (enabled_) {
    CNPERF_CALL(cnperfConfigSet(
          config,
          "device_id",
          &dev_id_,
          sizeof(dev_id_)));
    CNPERF_CALL(cnperfConfigSet(config,
          "trace_kernel_pmu",
          &enabled_,
          sizeof(enabled_)));
    CNPERF_CALL(cnperfConfigSet(config,
          "kernel_pmu_event_traced",
          SELECTED_PMU_EVENTS,
          strlen(SELECTED_PMU_EVENTS) + 1));
  }
}

CnperfPmuApi::CnperfPmuApi() {
  static bool enable_pmu = [] {
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
  cnrtGetDevice(&dev_id_);
  enabled_ = enable_pmu;
}

void CnperfPmuApi::recordPmuData(
    uint64_t corrid,
    CnperfPmuData pmu_data) {
  if (pmu_datamap_) {
    if (auto search = pmu_datamap_->find(corrid); search != pmu_datamap_->end()) {
      pmu_datamap_->at(corrid).emplace_back(pmu_data);
    } else {
      pmu_datamap_->emplace(corrid, std::vector<CnperfPmuData>{pmu_data,});
    }
  } else {
    pmu_datamap_ = std::make_unique<PmuDataMap>();
    pmu_datamap_->emplace(corrid, std::vector<CnperfPmuData>{pmu_data,});
  }
}

}
