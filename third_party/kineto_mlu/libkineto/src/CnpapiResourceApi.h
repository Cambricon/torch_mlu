#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <stack>
#include <memory>
#include <cnpapi.h>

#include "cnpapi_call.h"
#include "CnpapiCallbackManager.h"
#include "ITraceActivity.h"


namespace KINETO_NAMESPACE {

using namespace libkineto;

class CnpapiResourceApi {
  public:
    static CnpapiResourceApi& singleton();

    void pushCorrelationID(int64_t id) {
      if (enabled_) {
        external_id_stack_.push(id);
      }
    }
    void popCorrelationID() {
      if (enabled_) {
        external_id_stack_.pop();
      }
    }

    bool enabled() const { return enabled_; }

    void processTaskTopoData(
      const std::unordered_map<int64_t, const ITraceActivity*>& cpu_activities);

    std::string getExternalOpName(uint64_t entity_node_id) {
        if (enabled_ && tasktopo_entity_to_op_name_.find(entity_node_id)
                        != tasktopo_entity_to_op_name_.end()) {
          return tasktopo_entity_to_op_name_.at(entity_node_id);
        } else {
          return std::string();
        }
    }

    static void recordTaskTopoData(
      void *userdata,
      cnpapi_CallbackDomain domain,
      cnpapi_CallbackId cbid,
      const cnpapi_ResourceData *cbdata);

  private:
    CnpapiResourceApi();
    ~CnpapiResourceApi() = default;

    // The high 32 bits represent the tasktopo id,
    // and the low 32 bits represent the node id.
    uint64_t getEntityNodeId(uint64_t topo_node_id) {
      uint32_t topo_id = static_cast<uint32_t>(topo_node_id >> 32);
      uint32_t node_id = static_cast<uint32_t>(topo_node_id & 0xFFFFFFFF);
      return (static_cast<uint64_t>(tasktopo_to_entity_.at(topo_id)) << 32) | node_id;
    }

    bool enabled_ = false;

    std::stack<int64_t> external_id_stack_;
    std::unordered_map<uint64_t, int64_t> tasktopo_node_to_external_id_;
    std::unordered_map<uint32_t, uint32_t> tasktopo_to_entity_;
    std::unordered_map<uint64_t, std::string> tasktopo_entity_to_op_name_;
};

struct CnpapiResourceCallbackRegistration {
  CnpapiResourceCallbackRegistration();
};

} // namespace KINETO_NAMESPACE
