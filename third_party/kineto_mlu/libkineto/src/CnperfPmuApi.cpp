#include "CnperfPmuApi.h"

#include <cnrt.h>
#include <cstring>

namespace KINETO_NAMESPACE {

CnperfPmuApi& CnperfPmuApi::singleton() {
  static CnperfPmuApi singleton_;
  return singleton_;
}

void CnperfPmuApi::init(const std::string& config) {
  CnperfPmuConfig pmu_cfg;
  pmu_cfg.parse(config);
  enabled_ = pmu_cfg.valid();
  selected_pmu_counters_ = pmu_cfg.getCounterString();
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
          selected_pmu_counters_.c_str(),
          selected_pmu_counters_.size() + 1));
  }
}

CnperfPmuApi::CnperfPmuApi() {
  cnrtGetDevice(&dev_id_);
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
