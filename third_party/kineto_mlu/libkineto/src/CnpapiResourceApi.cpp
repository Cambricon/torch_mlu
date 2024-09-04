#include "CnpapiResourceApi.h"

#include <algorithm>

namespace KINETO_NAMESPACE {

CnpapiResourceApi& CnpapiResourceApi::singleton() {
    static CnpapiResourceApi singleton;
    return singleton;
}

CnpapiResourceApi::CnpapiResourceApi() : enabled_(false) {
  static bool enable_resource_callback = [] {
    bool result = false;
    const auto env = std::getenv("TORCH_MLU_ENABLE_CATCHING_MLUGRAPH_OP");
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
  if (enable_resource_callback) {
      enabled_ = true;
  }
}

void CnpapiResourceApi::recordTaskTopoData(
    void *userdata,
    cnpapi_CallbackDomain domain,
    cnpapi_CallbackId cbid,
    const cnpapi_ResourceData* cbdata) {
  if (domain == CNPAPI_CB_DOMAIN_RESOURCE) {
    if (cbid == CNPAPI_RESOURCE_TRACE_CBID_TASKTOPO_NODE_CREATED &&
        !singleton().external_id_stack_.empty()) {
      auto tasktopo_data = reinterpret_cast<cnpapi_TaskTopoData*>(cbdata->resource_ptr);
      uint64_t node_id;
      CNPAPI_CALL(cnpapiGetCNtaskTopoNodeId(tasktopo_data->node, &node_id));
      int64_t external_id = singleton().external_id_stack_.top();
      singleton().tasktopo_node_to_external_id_.insert({node_id, external_id});
    } else if (cbid == CNPAPI_RESOURCE_TRACE_CBID_TASKTOPO_ENTITY_CREATE_STARTING) {
      auto tasktopo_data = reinterpret_cast<cnpapi_TaskTopoData*>(cbdata->resource_ptr);
      uint32_t topo_id;
      uint32_t entity_id;
      CNPAPI_CALL(cnpapiGetCNtaskTopoId(tasktopo_data->tasktopo, &topo_id));
      CNPAPI_CALL(cnpapiGetCNtaskTopoEntityId(tasktopo_data->tasktopo_entity, &entity_id));
      singleton().tasktopo_to_entity_.insert({topo_id, entity_id});
    }
  }
}

void CnpapiResourceApi::processTaskTopoData(
    const std::unordered_map<int64_t, const ITraceActivity*>& cpu_activities) {
  if (enabled_) {
    for (const auto& node_to_ext_id : tasktopo_node_to_external_id_) {
      tasktopo_entity_to_op_name_.emplace(
          getEntityNodeId(node_to_ext_id.first),
          cpu_activities.at(node_to_ext_id.second)->name());
    }
    tasktopo_node_to_external_id_.clear();
    tasktopo_to_entity_.clear();
  }
}

CnpapiResourceCallbackRegistration::CnpapiResourceCallbackRegistration() {
  if (CnpapiResourceApi::singleton().enabled()) {
    kineto_mlu::CnpapiCallbackManager::getInstance()
      .registerCallbackFunction((cnpapi_CallbackFunc)CnpapiResourceApi::recordTaskTopoData)
      .insertEnabledCbid(CNPAPI_CB_DOMAIN_RESOURCE,
                         CNPAPI_RESOURCE_TRACE_CBID_TASKTOPO_NODE_CREATED)
      .insertEnabledCbid(CNPAPI_CB_DOMAIN_RESOURCE,
                         CNPAPI_RESOURCE_TRACE_CBID_TASKTOPO_ENTITY_CREATE_STARTING);
  }
};

}  // namespace KINETO_NAMESPACE
