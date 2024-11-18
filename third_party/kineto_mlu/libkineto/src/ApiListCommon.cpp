#include "ApiListCommon.h"

#include "cnperf_call.h"
#include "Logger.h"

#include <algorithm>
#include <fstream>
#include <regex>
#include <unordered_set>

#include "nlohmann/json.hpp"

namespace KINETO_NAMESPACE {

using namespace libkineto;

const std::vector<std::string> enabledCnrtApiList = {
  "cnrtMalloc",
  "cnrtMallocBatch",
  "cnrtMallocHost",
  "cnrtMemcpy",
  "cnrtMemcpyBatch",
  "cnrtMemcpyBatchByDesc",
  "cnrtMemcpyBatchByDescArray",
  "cnrtMemcpyByDesc",
  "cnrtMemcpyByDescArray",
  "cnrtMemset",
  "cnrtMemcpyPeer",
  "cnrtMemcpyAsync",
  "cnrtMemsetD8",
  "cnrtMemsetD32",
  "cnrtMemsetD8Async",
  "cnrtMemsetD32Async",
  "cnrtSyncDevice",
  "cnrtMemcpyPeerAsync",
  "cnrtHostMalloc",
  "cnrtQueueCreate",
  "cnrtQueueDestroy",
  "cnrtQueueQuery",
  "cnrtQueueSync",
  "cnrtMemsetAsync",
  "cnrtMemcpy2D",
  "cnrtMemcpy3D",
  "cnrtMemcpyAsync_V2",
};

const std::vector<std::string> enabledCndrvApiList = {
  "cnInvokeKernel",
  "cnInvokeKernelEx",
  "cnTaskTopoEntityInvoke",
  "cnPlaceNotifier",
  "cnPlaceNotifierWithFlags",
  "cnQueueWaitNotifier",
  "cnQueueWaitNotifierWithFlags",
  "cnMemcpyHtoDAsync",
  "cnMemcpyHtoDAsync_V2",
  "cnMemcpyDtoHAsync",
  "cnMemcpyDtoHAsync_V2",
  "cnMemcpyPeerAsync",
  "cnMemcpyDtoDAsync",
  "cnMemcpyAsync",
  "cnMemcpyAsync_V2",
  "cnMemcpy",
  "cnMemcpy2D",
  "cnMemcpy2DAsync",
  "cnMemcpy3D",
  "cnMemcpy3DAsync",
  "cnMemcpyPeer",
  "cnMemcpyHtoD",
  "cnMemcpyDtoH",
  "cnMemcpyDtoD",
  "cnMemcpyDtoD2D",
  "cnMemcpyDtoD3D",
  "cnMemsetD8Async",
  "cnMemsetD16Async",
  "cnMemsetD32Async",
  "cnMemsetD8",
  "cnMemsetD16",
  "cnMemsetD32",
  "cnInvokeHostFunc",
  "cnCnpapiInternalReserved1",
  "cnQueueAtomicOperation",
  "cnMalloc",
  "cnMallocSecurity",
  "cnMallocNode",
  "cnZmalloc",
  "cnZmallocNode",
  "cnMallocConstant",
  "cnMallocNodeConstant",
  "cnMallocFrameBuffer",
  "cnMallocPeerAble",
  "cnFree",
  "cnNCSLaunchKernel",
};

const std::vector<std::string> disabledCnnlApiPatternList = {
  "^cnnl.*WorkspaceSize.*",
  "^cnnl.*Descriptor.*",
  "^cnnl.*DescCreate.*",
  "^cnnl.*DescDestroy.*",
  "^cnnl.*DescAttr.*",
  "^cnnl.*AlgoCreate.*",
  "^cnnl.*AlgoDestroy.*",
  "^cnnlSet.*",
  "^cnnlGet.*",
  "^cnnlCreate.*",
  "^cnnlDestroy.*",
};

const std::vector<std::string> enabledDefaultAllPatternList = {
  "^cnnl.*",
  "^cncl.*",
  "^cnpx.*",
};

static size_t getTotalSize(const std::vector<std::string>& str_list) {
  size_t total_size = 0;
  for (const auto& str : str_list) {
    total_size += str.size();
    // Plus the size of a comma
    total_size += 1;
  }
  return total_size;
}

static void combineString(const std::vector<std::string>& str_list,
                          std::string* result_str) {
  for (int i = 0; i < str_list.size(); ++i) {
    if (!result_str->empty()) {
      *result_str += ",";
    }
    *result_str += str_list[i];
  }
}

bool ApiList::initializeJsonConfig() {
  const char* json_config_env = std::getenv("KINETO_MLU_USE_CONFIG");
  if (json_config_env && *json_config_env) {
    jsonConfigPath_ = json_config_env;
    return true;
  }
  return false;
}

bool ApiList::recordAllApis() {
  bool result = false;
  const auto record_env = std::getenv("KINETO_MLU_RECORD_ALL_APIS");
  if (record_env) {
    std::string record_env_str = record_env;
    std::transform(record_env_str.begin(), record_env_str.end(), record_env_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (record_env_str == "true" || record_env_str == "1" || record_env_str == "on") {
      result = true;
    }
  }
  return result;
}

// If 'all' in some white or black lists, replace it to target_str and clear other values.
void ApiList::replaceAllValue(
    std::vector<std::string>* str_list, const std::string& target_str) {
  if (std::find(str_list->begin(), str_list->end(), "all") != str_list->end()) {
    str_list->clear();
    str_list->push_back(target_str);
  }
}

void ApiList::updateBlackListWithWhiteList(
    const std::vector<std::string>& white_list,
    std::vector<std::string>* black_list) {
  if (std::find(white_list.begin(), white_list.end(), cnnlAllPattern_) != white_list.end()) {
    // If white list record all, we should set black list to empty.
    black_list->clear();
    return;
  }
  for (const std::string& str : white_list) {
    for (size_t i = 0; i < black_list->size(); ++i) {
      std::regex pattern((*black_list)[i]);
      if (std::regex_match(str, pattern)) {
        (*black_list)[i] = "(?!^" + str + "$)" + (*black_list)[i];
      }
    }
  }
}

// Static method to get the singleton instance
ApiList& ApiList::getInstance() {
    static ApiList instance;
    return instance;
}

ApiList::ApiList() {
  useConfig_ = ApiList::initializeJsonConfig();
  recordAll_ = ApiList::recordAllApis();
  if (recordAll_) {
    enabled_list_str_ = std::string(".*");
    disabled_list_str_ = std::string("");
    return;
  }

  // Init white lists using default config.
  size_t white_list_total_size =
      getTotalSize(enabledDefaultAllPatternList) +
      getTotalSize(enabledCnrtApiList) +
      getTotalSize(enabledCndrvApiList);
  enabled_list_str_.reserve(white_list_total_size);
  combineString(enabledDefaultAllPatternList, &enabled_list_str_);
  combineString(enabledCnrtApiList, &enabled_list_str_);
  combineString(enabledCndrvApiList, &enabled_list_str_);

  size_t black_list_total_size = getTotalSize(disabledCnnlApiPatternList);
  if (!useConfig_) {
    // Init black lists using default config.
    disabled_list_str_.reserve(black_list_total_size);
    combineString(disabledCnnlApiPatternList, &disabled_list_str_);
  } else {
    // Update black and white lists if using custom config.
    // Default cnnl disabled pattern should exclude the patterns configured
    // in custom cnnl_whitelist so that we could catch these apis.
    std::vector<std::string> new_disabled_cnnl_pattern_list = disabledCnnlApiPatternList;

    std::ifstream inputFile(jsonConfigPath_);
    if (!inputFile.is_open()) {
      LOG(ERROR) << "Could not open the file: " << jsonConfigPath_;
      return;
    }
    nlohmann::json jsonObject;
    inputFile >> jsonObject;

    if (jsonObject.contains("cnnl_blacklist")) {
      cnnlBlackList_ = jsonObject["cnnl_blacklist"];
      replaceAllValue(&cnnlBlackList_, cnnlAllPattern_);
    }
    if (jsonObject.contains("cnnl_whitelist")) {
      cnnlWhiteList_ = jsonObject["cnnl_whitelist"];
      replaceAllValue(&cnnlWhiteList_, cnnlAllPattern_);
      updateBlackListWithWhiteList(cnnlWhiteList_, &new_disabled_cnnl_pattern_list);
    }
    if (jsonObject.contains("cnrt_blacklist")) {
      cnrtBlackList_ = jsonObject["cnrt_blacklist"];
      replaceAllValue(&cnrtBlackList_, cnrtAllPattern_);
    }
    if (jsonObject.contains("cnrt_whitelist")) {
      cnrtWhiteList_ = jsonObject["cnrt_whitelist"];
      replaceAllValue(&cnrtWhiteList_, cnrtAllPattern_);
    }
    if (jsonObject.contains("cndrv_blacklist")) {
      cndrvBlackList_ = jsonObject["cndrv_blacklist"];
      replaceAllValue(&cndrvBlackList_, cndrvAllPattern_);
    }
    if (jsonObject.contains("cndrv_whitelist")) {
      cndrvWhiteList_ = jsonObject["cndrv_whitelist"];
      replaceAllValue(&cndrvWhiteList_, cndrvAllPattern_);
    }
    white_list_total_size +=
        getTotalSize(cnnlWhiteList_) +
        getTotalSize(cnrtWhiteList_) +
        getTotalSize(cndrvWhiteList_);
    black_list_total_size =
        getTotalSize(new_disabled_cnnl_pattern_list) +
        getTotalSize(cnnlBlackList_) +
        getTotalSize(cnrtBlackList_) +
        getTotalSize(cndrvBlackList_);
    enabled_list_str_.reserve(white_list_total_size);
    disabled_list_str_.reserve(black_list_total_size);
    combineString(cnnlWhiteList_, &enabled_list_str_);
    combineString(cnrtWhiteList_, &enabled_list_str_);
    combineString(cndrvWhiteList_, &enabled_list_str_);
    combineString(new_disabled_cnnl_pattern_list, &disabled_list_str_);
    combineString(cnnlBlackList_, &disabled_list_str_);
    combineString(cnrtBlackList_, &disabled_list_str_);
    combineString(cndrvBlackList_, &disabled_list_str_);
  }
}

void ApiList::updateConfig(cnperfConfig_t config) {
  CNPERF_CALL(cnperfConfigSet(
      config,
      "cambricon_api_traced",
      enabled_list_str_.c_str(),
      enabled_list_str_.size() + 1));
  CNPERF_CALL(cnperfConfigSet(
      config,
      "cambricon_api_untraced",
      disabled_list_str_.c_str(),
      disabled_list_str_.size() + 1));
}

} // namespace KINETO_NAMESPACE
