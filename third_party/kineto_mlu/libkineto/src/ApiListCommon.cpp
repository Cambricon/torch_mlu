#include "CnpapiActivityApi.h"
#include "Logger.h"
#include "cnpapi_strings.h"
#include "ApiListCommon.h"
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include "nlohmann/json.hpp"

namespace KINETO_NAMESPACE {

const std::vector<cnpapi_CallbackIdCNRT> enabledCnrtCbidList_ = {
  CNPAPI_CNRT_TRACE_CBID_cnrtMalloc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMallocBatch,
  CNPAPI_CNRT_TRACE_CBID_cnrtMallocHost,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatch,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatchByDesc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyBatchByDescArray,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyByDesc,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyByDescArray,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemset,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyPeer,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD8,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD32,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD8Async,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetD32Async,
  CNPAPI_CNRT_TRACE_CBID_cnrtSyncDevice,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyPeerAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtHostMalloc,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueCreate,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueDestroy,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueQuery,
  CNPAPI_CNRT_TRACE_CBID_cnrtQueueSync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemsetAsync,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy2D,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpy3D,
  CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyAsync_V2
};

const std::vector<cnpapi_CallbackIdCNDRV> enabledCndrvCbidList_ = {
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel,
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernelEx,
  CNPAPI_CNDRV_TRACE_CBID_cnTaskTopoEntityInvoke,
  CNPAPI_CNDRV_TRACE_CBID_cnPlaceNotifier,
  CNPAPI_CNDRV_TRACE_CBID_cnPlaceNotifierWithFlags,
  CNPAPI_CNDRV_TRACE_CBID_cnQueueWaitNotifier,
  CNPAPI_CNDRV_TRACE_CBID_cnQueueWaitNotifierWithFlags,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoDAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoDAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoHAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoHAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyPeerAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoDAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync_V2,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy2D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy2DAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy3D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpy3DAsync,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyPeer,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyHtoD,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoH,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD2D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemcpyDtoD3D,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD8Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD16Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32Async,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD8,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD16,
  CNPAPI_CNDRV_TRACE_CBID_cnMemsetD32,
  CNPAPI_CNDRV_TRACE_CBID_cnInvokeHostFunc,
  CNPAPI_CNDRV_TRACE_CBID_cnCnpapiInternalReserved1,
  CNPAPI_CNDRV_TRACE_CBID_cnQueueAtomicOperation,
  CNPAPI_CNDRV_TRACE_CBID_cnMalloc,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocSecurity,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocNode,
  CNPAPI_CNDRV_TRACE_CBID_cnZmalloc,
  CNPAPI_CNDRV_TRACE_CBID_cnZmallocNode,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocConstant,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocNodeConstant,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocFrameBuffer,
  CNPAPI_CNDRV_TRACE_CBID_cnMallocPeerAble,
  CNPAPI_CNDRV_TRACE_CBID_cnFree
};

bool ApiList::initializeJsonConfig() {
    const char* json_config_env = std::getenv("KINETO_MLU_USE_CONFIG");
    return json_config_env && *json_config_env;
}

// Static method to get the singleton instance
ApiList& ApiList::getInstance() {
    static ApiList instance;
    return instance;
}

ApiList::ApiList() {
    useConfig_ = ApiList::initializeJsonConfig();
    // Initialize cnnl enable list
    std::vector<std::string> disable_match_ptr = {"WorkspaceSize",
                                                  "Descriptor",
                                                  "DescCreate",
                                                  "DescDestroy",
                                                  "DescAttr",
                                                  "AlgoCreate",
                                                  "AlgoDestroy",
                                                  "cnnlSet",
                                                  "cnnlGet",
                                                  "cnnlCreate",
                                                  "cnnlDestroy"};
    auto isMatched = [&](const std::string& name) {
      for (const auto& str : disable_match_ptr) {
        if (name.find(str) != std::string::npos) {
          return true;
        }
      }
      return false;
    };
    for (int i = 0; i < CNPAPI_CNNL_TRACE_CBID_SIZE; ++i) {
      cnpapi_CallbackIdCNNL cbid = static_cast<cnpapi_CallbackIdCNNL>(i);
      std::string name = runtimeCbidName(CNPAPI_ACTIVITY_TYPE_CNNL_API, cbid);
      if (!isMatched(name)) {
        enabledCnnlCbidList_.push_back(cbid);
      }
    }
    // Update black and white lists if using custom config
    if (useConfig_) {
        const std::string json_config_path = std::getenv("KINETO_MLU_USE_CONFIG");
	std::ifstream inputFile(json_config_path);
	if (!inputFile.is_open()) {
            LOG(ERROR) << "Could not open the file: " << json_config_path;
            return;
        }
	nlohmann::json jsonObject;
	inputFile >> jsonObject;

	if (jsonObject.contains("cnnl_blacklist")) {
          cnnlBlackList_ = jsonObject["cnnl_blacklist"];
	}
	if (jsonObject.contains("cnnl_whitelist")) {
          cnnlWhiteList_ = jsonObject["cnnl_whitelist"];
	}
	if (jsonObject.contains("cnrt_blacklist")) {
          cnrtBlackList_ = jsonObject["cnrt_blacklist"];
	}
	if (jsonObject.contains("cnrt_whitelist")) {
          cnrtWhiteList_ = jsonObject["cnrt_whitelist"];
	}
	if (jsonObject.contains("cndrv_blacklist")) {
          cndrvBlackList_ = jsonObject["cndrv_blacklist"];
	}
	if (jsonObject.contains("cndrv_whitelist")) {
          cndrvWhiteList_ = jsonObject["cndrv_whitelist"];
	}
    }
}

template<typename CallbackIdType, int TRACE_CBID_SIZE>
std::vector<CallbackIdType> getCbidList(const std::vector<std::string>& blacklist, 
                                        const std::vector<std::string>& whitelist, 
                                        cnpapiActivityType activityType,
					const std::vector<CallbackIdType>& orig_list) {
    std::vector<CallbackIdType> enabled_list = orig_list;

    // Check if whitelist contains "all"
    bool all_in_whitelist = (std::find(whitelist.begin(), whitelist.end(), "all") != whitelist.end());

    // If whitelist contains "all", return all callback IDs
    if (all_in_whitelist) {
        for (int i = 0; i < TRACE_CBID_SIZE; ++i) {
            CallbackIdType cbid = static_cast<CallbackIdType>(i);
            // Check if cbid is not already in enabled_list
            if (std::find(enabled_list.begin(), enabled_list.end(), cbid) == enabled_list.end()) {
                enabled_list.push_back(cbid);
            }
        }
        return enabled_list;
    }
    // If blacklist contains "all", return empty list
    if (std::find(blacklist.begin(), blacklist.end(), "all") != blacklist.end()) {
        return {};
    }

    // Add whitelisted items to enabled_list
    for (const auto& str : whitelist) {
        for (int i = 0; i < TRACE_CBID_SIZE; ++i) {
            CallbackIdType cbid = static_cast<CallbackIdType>(i);
            std::string name = runtimeCbidName(activityType, cbid);
            if (name.find(str) != std::string::npos && 
                std::find(enabled_list.begin(), enabled_list.end(), cbid) == enabled_list.end()) {
                enabled_list.push_back(cbid);
            }
        }
    }

    // Remove blacklisted items from enabled_list
    enabled_list.erase(
        std::remove_if(enabled_list.begin(), enabled_list.end(), [&](const CallbackIdType& cbid) {
            std::string name = runtimeCbidName(activityType, cbid);
            return std::find(blacklist.begin(), blacklist.end(), name) != blacklist.end();
        }), 
        enabled_list.end()
    );

    return enabled_list;
}

const std::vector<cnpapi_CallbackIdCNNL> ApiList::getCnnlCbidList() {
    return getCbidList<cnpapi_CallbackIdCNNL, CNPAPI_CNNL_TRACE_CBID_SIZE>(cnnlBlackList_,
	cnnlWhiteList_, CNPAPI_ACTIVITY_TYPE_CNNL_API, enabledCnnlCbidList_);
}

const std::vector<cnpapi_CallbackIdCNRT> ApiList::getCnrtCbidList() {
    return getCbidList<cnpapi_CallbackIdCNRT, CNPAPI_CNRT_TRACE_CBID_SIZE>(cnrtBlackList_,
	cnrtWhiteList_, CNPAPI_ACTIVITY_TYPE_CNRT_API, enabledCnrtCbidList_);
}

const std::vector<cnpapi_CallbackIdCNDRV> ApiList::getCndrvCbidList() {
    return getCbidList<cnpapi_CallbackIdCNDRV, CNPAPI_CNDRV_TRACE_CBID_SIZE>(cndrvBlackList_,
	cndrvWhiteList_, CNPAPI_ACTIVITY_TYPE_CNDRV_API, enabledCndrvCbidList_);
}

const std::vector<cnpapi_CallbackIdCNDRV>& ApiList::getCndrvLaunchCbidList() {
    return enabledCndrvCbidList_;
}

} // namespace KINETO_NAMESPACE
