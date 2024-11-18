#pragma once
#include <cnperf_api.h>
#include <cnperf_call.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace KINETO_NAMESPACE {

struct CnperfPmuData {
  std::string counter_name;
  // pmu value have 3 type, see cnperfDataPmuValue_t
  std::variant<uint64_t, int64_t, double> value;
};

// [correlation_id -> [CnperfPmuData]]
using PmuDataMap = std::unordered_map<uint64_t, std::vector<CnperfPmuData>>;

class CnperfPmuApi {
  public:
    static CnperfPmuApi& singleton();

    bool enabled() const { return enabled_; }

    void updateConfig(cnperfConfig_t config);

    void recordPmuData(uint64_t corrid, CnperfPmuData pmu_data);

    std::unique_ptr<PmuDataMap> getPmuData() { return std::move(pmu_datamap_); }

  private:
    CnperfPmuApi();

    int dev_id_;
    bool enabled_ = false;
    std::unique_ptr<PmuDataMap> pmu_datamap_;
};

}  // namespace KINETO_NAMESPACE
