#include "CnpapiCallbackManager.h"


namespace kineto_mlu {

CnpapiCallbackManager& CnpapiCallbackManager::getInstance() {
    static CnpapiCallbackManager instance;
    return instance;
}

CnpapiCallbackManager& CnpapiCallbackManager::registerCallbackFunction(cnpapi_CallbackFunc func) {
    callback_funcs_.emplace_back(std::make_unique<cnpapi_CallbackFunc>(func));
    return *this;
}

CnpapiCallbackManager& CnpapiCallbackManager::insertEnabledCbid(cnpapi_CallbackDomain domain,
                                              cnpapi_CallbackId cbid) {
    auto it = enabled_cbids_.find(domain);
    if (it != enabled_cbids_.end()) {
        it->second.insert(cbid);
    } else {
        enabled_cbids_[domain].insert(cbid);
    }
    return *this;
}

}  // kineto_mlu
