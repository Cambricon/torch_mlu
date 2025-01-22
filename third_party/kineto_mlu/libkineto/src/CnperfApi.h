#pragma once

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <cnperf_api.h>

#include "cnperf_call.h"
#include "ActivityType.h"
#include "CnperfRecord.h"
#include "ITraceActivity.h"
#include "TimeGapCleaner.h"
#include "CnperfPmuApi.h"


namespace KINETO_NAMESPACE {

using namespace libkineto;

class CnperfApi {
  public:
    enum CorrelationFlowType {
        Default,
        User
    };

    CnperfApi();
    CnperfApi(const CnperfApi&) = delete;
    CnperfApi& operator=(const CnperfApi&) = delete;

    virtual ~CnperfApi() {
      CNPERF_CALL(cnperfRelease());
    }

    static CnperfApi& singleton();

    void prepare();
    void start();
    void stop();
    void process(const std::unordered_map<int64_t, const ITraceActivity*>& cpu_activities);

    std::unique_ptr<CnperfRecordSet> getAllRecords() {
      return std::move(all_records_);
    }

    std::unique_ptr<std::unordered_set<uint64_t>> getDeviceTaskCorrelations() {
      return std::move(device_task_correlations_);
    }

    const std::unordered_map<uint64_t, uint64_t>& getCpuCorrelationMap() {
      return cpu_correlation_map_;
    }

    const std::unordered_map<uint64_t, uint64_t>& getUserCorrelationMap() {
      return user_correlation_map_;
    }

    static void pushCorrelationID(int id, CorrelationFlowType type);
    static void popCorrelationID(CorrelationFlowType type);

    void clearTraceData();

  private:
    int dev_id_;
    cnperfSession_t session_;
    cnperfConfig_t config_;
    cnperfParserHandle_t parser_;
    uint64_t cnperf_start_ts_;
    bool enable_catching_mlugraph_op_;

    std::unordered_set<uint64_t> user_external_ids_;
    std::unordered_map<uint64_t, uint64_t> cpu_correlation_map_;
    std::unordered_map<uint64_t, uint64_t> user_correlation_map_;
    std::unique_ptr<CnperfRecordSet> all_records_;
    // Used to determine whether runtime api is flow event start.
    std::unique_ptr<std::unordered_set<uint64_t>> device_task_correlations_;

    void fillCpuActivities(
        const std::unordered_map<int64_t, const ITraceActivity*>& cpu_activities);
    static void processFunction(cnperfDataFunction_t* data, void* userdata);
    static void processDeviceTask(cnperfDataDeviceTask_t* data, void* userdata);
    static void saveExternalId(cnperfDataOpRange_t* data,
        std::vector<std::pair<uint64_t, uint64_t>>* time_to_external_id);
    static void saveTasktopoExternalOp(cnperfDataOpRange_t* data,
        std::vector<std::pair<uint64_t, std::pair<uint64_t, std::string>>>* time_to_external_op);
    static void savePmuData(const cnperfDataPmuInfo_t* pmu_info, 
        const cnperfDataPmuValue_t* value, uint64_t* corrid);
    static void processCommData(
        const cnperfDataOpRange_t* op_range,
        const cnperfDataCommTask_t* data,
        void* userdata);
};

} // namespace KINETO_NAMESPACE
