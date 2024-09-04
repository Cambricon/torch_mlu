#include "CnpapiPmuApi.h"

#include <algorithm>
#include <cnrt.h>
#include <cnpapi_generated_cndrv_params.h>

#include "Logger.h"
#include "ThreadUtil.h"
#include "CnpapiCallbackManager.h"
// TODO(fuwenguang): Delete cnpapi_addition.h
// See SYSTOOL-4476
#include "cnpapi_addition.h"

#include "Logger.h"
#include "ThreadUtil.h"

namespace KINETO_NAMESPACE {

namespace {
const std::vector<std::string> SELECTED_COUNTER_NAMES = {
    "llc__tagram_hit",
    "llc__tagram_miss",
    "llc__eviction",
    "tp_core__lt_cycles",
    "tp_core__csimd_post_cycles",
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
    "llc__eviction"
};
}  // namespace

CnpapiPmuApi& CnpapiPmuApi::singleton() {
    static CnpapiPmuApi singleton;
    return singleton;
}

CnpapiPmuApi::CnpapiPmuApi() {
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
  if (enable_pmu) {
    cnrtGetDevice(&devId_);
    CNPAPI_CALL(cnpapiGetDeviceChipId(devId_, &chipId_));
    auto status = CNPAPI_CALL_NOWARN(cnpapiPmuInit(devId_));
    if (status == CNPAPI_SUCCESS) {
      CNPAPI_CALL(cnpapiPmuSetFlushMode(CNPAPI_PMU_EXPLICIT_FLUSH));
      enableCounters();
      enabled_ = true;
      return;
    }
    LOG(ERROR) << "CNPapi Pmu initialization failed. "
               << "Pmu Data will be missing. "
               << "Device Id: " << devId_ << ". "
               << "Process Id: " << processId() << ".";
  }
  enabled_ = false;
}

void CnpapiPmuApi::catchingPmuData(void *userdata, cnpapi_CallbackDomain domain,
                                   cnpapi_CallbackId cbid, const cnpapi_CallbackData *cbdata) {
  if (domain == CNPAPI_CB_DOMAIN_CNDRV_API &&
      cbid == CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel) {
    CNkernel kernel = reinterpret_cast<const cnpapi_cnInvokeKernel_params *>(
      cbdata->functionParams)->hkernel;
    uint8_t skip_record = 0;
    // TODO(fuwenguang): Replace cnpapiIsKernelTypeTCDP to the api exported by cnpapi.
    // See SYSTOOL-4476
    CNPAPI_CALL(cnpapiIsKernelTypeTCDP(kernel, &skip_record));
    if (skip_record) {
      return;
    }
    CNqueue hqueue = reinterpret_cast<const cnpapi_cnInvokeKernel_params *>(
      cbdata->functionParams)->hqueue;
    cnrtQueueCaptureStatus_t capture_status;
    cnrtQueueIsCapturing(hqueue, &capture_status);
    // Only record pmu data when the queue is not capturing.
    if (capture_status != cnrtQueueCaptureStatusNone) {
      return;
    }
    if (cbdata->callbackSite == CNPAPI_API_ENTER) {
      singleton().recordData(RecordStage::START, cbdata->correlationId, hqueue);
    } else if (cbdata->callbackSite == CNPAPI_API_EXIT) {
      singleton().recordData(RecordStage::END, cbdata->correlationId, hqueue);
    }
  }
}

void CnpapiPmuApi::recordData(RecordStage stage, uint64_t correlation_id, CNqueue hqueue) {
  if (stage == RecordStage::START) {
    corrIdToPmuDataStates_.emplace(correlation_id, std::unordered_map<uint64_t, CnpapiPmuDataState>{});
  } else if (stage == RecordStage::END) {
    cnrtQueueSync(hqueue);
  }
  CNPAPI_CALL(cnpapiPmuFlushData(devId_));
  for (const auto& counter : enabledCounters_) {
    uint64_t value = 0;
    CNPAPI_CALL(cnpapiPmuGetCounterValue(devId_, counter.first, &value));
    if (stage == RecordStage::START) {
      corrIdToPmuDataStates_[correlation_id].emplace(counter.first, CnpapiPmuDataState{0, 0});
      corrIdToPmuDataStates_[correlation_id][counter.first].start_value = value;
    } else if (stage == RecordStage::END) {
      corrIdToPmuDataStates_[correlation_id][counter.first].end_value = value;
    }
  }
}

void CnpapiPmuApi::enableCounters() {
  // get supported counter number
  uint64_t counter_num = 0;
  cnpapiPmuGetCounterSupported(chipId_, nullptr, &counter_num);
  // get supported counter id
  std::vector<uint64_t> counter_ids(counter_num);
  CNPAPI_CALL(cnpapiPmuGetCounterSupported(chipId_, counter_ids.data(), &counter_num));

  for (auto counter_id : counter_ids) {
    const char *counter_name;
    CNPAPI_CALL(cnpapiPmuGetCounterName(counter_id, &counter_name));
    for (const auto &name : SELECTED_COUNTER_NAMES) {
      if (name.compare(counter_name) == 0) {
        CNPAPI_CALL(cnpapiPmuEnableCounter(devId_, counter_id, true));
        enabledCounters_.emplace(counter_id, name);
        break;
      }
    }
  }
}

std::unique_ptr<PmuDataMap> CnpapiPmuApi::getPmuData() {
  if (corrIdToPmuDataStates_.empty()) {
    return nullptr;
  }
  std::unique_ptr<PmuDataMap> pmu_datas = std::make_unique<PmuDataMap>();
  for (const auto& corrid_state : corrIdToPmuDataStates_) {
    pmu_datas->emplace(corrid_state.first, std::vector<CnpapiPmuData>{});
    for (const auto& counterid_state : corrid_state.second) {
      const auto& counter_name = enabledCounters_[counterid_state.first];
      pmu_datas->at(corrid_state.first).emplace_back(
          CnpapiPmuData{counter_name, counterid_state.first,
            counterid_state.second.end_value - counterid_state.second.start_value});
    }
  }
  corrIdToPmuDataStates_.clear();
  return std::move(pmu_datas);
}

CnpapiPmuCallbackRegistration::CnpapiPmuCallbackRegistration() {
  if (CnpapiPmuApi::singleton().enabled()) {
    kineto_mlu::CnpapiCallbackManager::getInstance()
      .registerCallbackFunction((cnpapi_CallbackFunc)CnpapiPmuApi::catchingPmuData);
  }
};

}  // namespace KINETO_NAMESPACE
