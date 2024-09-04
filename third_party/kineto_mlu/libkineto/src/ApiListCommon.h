#pragma once
#include <vector>
#include <string>

namespace KINETO_NAMESPACE {

class ApiList {

public:
    static ApiList& getInstance();

    const std::vector<cnpapi_CallbackIdCNNL> getCnnlCbidList();
    const std::vector<cnpapi_CallbackIdCNRT> getCnrtCbidList();
    const std::vector<cnpapi_CallbackIdCNDRV> getCndrvCbidList();
    const std::vector<cnpapi_CallbackIdCNDRV>& getCndrvLaunchCbidList();

private:
    ApiList();
    ~ApiList() = default;

    // Delete copy constructor and assignment operator
    ApiList(const ApiList&) = delete;
    ApiList& operator=(const ApiList&) = delete;

    static bool initializeJsonConfig();

    static bool recordAllApis();
    
    bool useConfig_;

    bool recordAll_;
    
    std::vector<std::string> cnnlBlackList_;
    std::vector<std::string> cnnlWhiteList_;
    std::vector<std::string> cnrtBlackList_;
    std::vector<std::string> cnrtWhiteList_;
    std::vector<std::string> cndrvBlackList_;
    std::vector<std::string> cndrvWhiteList_;

    std::vector<cnpapi_CallbackIdCNNL> enabledCnnlCbidList_ = {};
};

} // namespace
