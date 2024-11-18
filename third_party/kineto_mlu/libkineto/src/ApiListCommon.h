#pragma once

#include <cnperf_api.h>

#include <vector>
#include <string>

namespace KINETO_NAMESPACE {

class ApiList {

public:
    static ApiList& getInstance();

    void updateConfig(cnperfConfig_t config);

private:
    ApiList();
    ~ApiList() = default;

    // Delete copy constructor and assignment operator
    ApiList(const ApiList&) = delete;
    ApiList& operator=(const ApiList&) = delete;

    bool initializeJsonConfig();
    bool recordAllApis();
    void replaceAllValue(
        std::vector<std::string>* str_list, const std::string& target_str);
    void updateBlackListWithWhiteList(
        const std::vector<std::string>& white_list,
        std::vector<std::string>* black_list);
    
    bool useConfig_;
    std::string jsonConfigPath_;
    bool recordAll_;
    
    std::vector<std::string> cnnlBlackList_;
    std::vector<std::string> cnnlWhiteList_;
    std::vector<std::string> cnrtBlackList_;
    std::vector<std::string> cnrtWhiteList_;
    std::vector<std::string> cndrvBlackList_;
    std::vector<std::string> cndrvWhiteList_;

    const std::string cnnlAllPattern_ = "^cnnl.*";
    const std::string cnrtAllPattern_ = "^cnrt.*";
    const std::string cndrvAllPattern_ = "^cn[A-Z].*";

    std::string enabled_list_str_;
    std::string disabled_list_str_;
};

}  // namespace KINETO_NAMESPACE
