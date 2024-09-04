#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <cnpapi.h>

#include "cnpapi_call.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;

struct CnpapiPmuData {
  std::string counter_name;
  uint64_t counter_id;
  uint64_t value;
};

// [correlation_id -> [CnpapiPmuData]]
using PmuDataMap = std::unordered_map<uint64_t, std::vector<CnpapiPmuData>>;

struct CnpapiPmuDataState {
  uint64_t start_value;
  uint64_t end_value;
};


class CnpapiPmuApi {
  public:
    static CnpapiPmuApi& singleton();

    bool enabled() const { return enabled_; }

    static void catchingPmuData(void *userdata, cnpapi_CallbackDomain domain,
                                cnpapi_CallbackId cbid, const cnpapi_CallbackData *cbdata);

    std::unique_ptr<PmuDataMap> getPmuData();

  private:
    CnpapiPmuApi();
    ~CnpapiPmuApi() {
      if (enabled_) {
        CNPAPI_CALL(cnpapiPmuRelease(devId_));
      }
    }

    enum class RecordStage {
      START = 0,
      END = 1,
    };

    void enableCounters();
    void recordData(RecordStage stage, uint64_t correlation_id, CNqueue hqueue);

    bool enabled_ = false;
    int devId_ = -1;
    uint64_t chipId_ = 0;

    std::unordered_map<uint64_t, std::string> enabledCounters_;

    // [counter_id -> CnpapiPmuDataState]
    using PmuStateMap = std::unordered_map<uint64_t, CnpapiPmuDataState>;
    // [correlation_id -> [counter_id -> CnpapiPmuDataState]]
    std::unordered_map<uint64_t, PmuStateMap> corrIdToPmuDataStates_;
};

struct CnpapiPmuCallbackRegistration {
  CnpapiPmuCallbackRegistration();
};

} // namespace KINETO_NAMESPACE
