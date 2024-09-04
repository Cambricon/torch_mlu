// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <atomic>
#ifdef HAS_CNPAPI
#include <cnpapi.h>
#endif
#include <array>
#include <list>
#include <memory>
#include <mutex>
#include <set>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CnpapiCallbackApiMock.h"

namespace KINETO_NAMESPACE {

using namespace libkineto;


/* CnpapiCallbackApi : Provides an abstraction over CNPAPI callback
 *  interface. This enables various callback functions to be registered
 *  with this class. The class registers a global callback handler that
 *  redirects to the respective callbacks.
 *
 *  Note: one design choice we made is to only support simple function pointers
 *  in order to speed up the implementation for fast path.
 */

using CnpapiCallbackFn = void(*)(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo);


class CnpapiCallbackApi {

 public:

  /* Global list of supported callback ids
   *  use the class namespace to avoid confusing with CNPAPI enums*/
  enum CnpapiCallBackID {
    CUDA_LAUNCH_KERNEL =  0,
    // can possibly support more callback ids per domain
    //
    __RUNTIME_CB_DOMAIN_START = CUDA_LAUNCH_KERNEL,

    // Callbacks under Resource CB domain
    RESOURCE_CONTEXT_CREATED,
    RESOURCE_CONTEXT_DESTROYED,

    __RUNTIME_CB_DOMAIN_END = RESOURCE_CONTEXT_CREATED,
    __RESOURCE_CB_DOMAIN_START = RESOURCE_CONTEXT_CREATED,

    __RESOURCE_CB_DOMAIN_END = RESOURCE_CONTEXT_DESTROYED + 1,
  };


  CnpapiCallbackApi(const CnpapiCallbackApi&) = delete;
  CnpapiCallbackApi& operator=(const CnpapiCallbackApi&) = delete;

  static CnpapiCallbackApi& singleton();

  bool initSuccess() const {
    return initSuccess_;
  }

#ifdef HAS_CNPAPI
  CUptiResult getCnpapiStatus() const {
    return lastCnpapiStatus_;
  }
#endif

  bool registerCallback(
    CUpti_CallbackDomain domain,
    CnpapiCallBackID cbid,
    CnpapiCallbackFn cbfn);

  // returns false if callback was not found
  bool deleteCallback(
    CUpti_CallbackDomain domain,
    CnpapiCallBackID cbid,
    CnpapiCallbackFn cbfn);

  bool enableCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid);
  bool disableCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid);


  // Please do not use this method. This has to be exposed as public
  // so it is accessible from the callback handler
  void __callback_switchboard(
      CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid,
      const CUpti_CallbackData* cbInfo);

 private:

  explicit CnpapiCallbackApi();

  // For callback table design overview see the .cpp file
  using CallbackList = std::list<CnpapiCallbackFn>;

  // level 2 tables sizes are known at compile time
  constexpr static size_t RUNTIME_CB_DOMAIN_SIZE
    = (__RUNTIME_CB_DOMAIN_END - __RUNTIME_CB_DOMAIN_START);

  constexpr static size_t RESOURCE_CB_DOMAIN_SIZE
    = (__RESOURCE_CB_DOMAIN_END - __RESOURCE_CB_DOMAIN_START);

  // level 1 table is a struct
  struct CallbackTable {
    std::array<CallbackList, RUNTIME_CB_DOMAIN_SIZE> runtime;
    std::array<CallbackList, RESOURCE_CB_DOMAIN_SIZE> resource;

    CallbackList* lookup(CUpti_CallbackDomain domain, CnpapiCallBackID cbid);
  };

  CallbackTable callbacks_;
  bool initSuccess_ = false;

#ifdef HAS_CNPAPI
  CUptiResult lastCnpapiStatus_;
  CUpti_SubscriberHandle subscriber_;
#endif
};

} // namespace KINETO_NAMESPACE
