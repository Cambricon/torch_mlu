#pragma once

#include <cnpapi.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace kineto_mlu {

using ENABLED_CBID_MAP =
std::unordered_map<cnpapi_CallbackDomain, std::unordered_set<cnpapi_CallbackId>>;

/**
 * @class CnpapiCallbackManager
 * @brief Manages the registration of CNPAPI callback functions and enabled cbids.
 * 
 * The CnpapiCallbackManager is a singleton class responsible for managing callback functions
 * and the associated enabled cbids for each callback domain.
 * It maintains a list of registered callback functions and a mapping between callback domains
 * and their enabled cbids. This class provides interfaces for registering callback functions,
 * inserting enabled cbids, and retrieving the stored callbacks and enabled cbids.
 */
class CnpapiCallbackManager {
  public:
    static CnpapiCallbackManager& getInstance();

    /**
     * @brief Registers a callback function.
     * @param func The callback function to register.
     * 
     * This method registers the provided callback function by storing it in the list of callback
     * functions. The function is wrapped in a std::unique_ptr and stored in the internal vector.
     */
    CnpapiCallbackManager& registerCallbackFunction(cnpapi_CallbackFunc func);

    /**
     * @brief Inserts a new cbid into the enabled cbid map.
     * @param domain The callback domain.
     * @param cbid The cbid to enable.
     * 
     * This method inserts the specified callback domain and cbid into the enabled cbid map.
     */
    CnpapiCallbackManager& insertEnabledCbid(cnpapi_CallbackDomain domain, cnpapi_CallbackId cbid);

    const std::vector<std::unique_ptr<cnpapi_CallbackFunc>>& getCallbackFunctions() {
      return callback_funcs_;
    }
    const ENABLED_CBID_MAP& getEnabledCbids() {
      return enabled_cbids_;
    }

  private:
    CnpapiCallbackManager() = default;

    std::vector<std::unique_ptr<cnpapi_CallbackFunc>> callback_funcs_;
    ENABLED_CBID_MAP enabled_cbids_;
};

}  // kineto_mlu
