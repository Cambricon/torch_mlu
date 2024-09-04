// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// Provides data structures to mock CNPAPI Callback API
#ifndef HAS_CNPAPI

enum CUpti_CallbackDomain {
  CNPAPI_CB_DOMAIN_RESOURCE,
  CNPAPI_CB_DOMAIN_RUNTIME_API,
};
enum CUpti_CallbackId {
  CNPAPI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
  CNPAPI_CBID_RESOURCE_CONTEXT_CREATED,
  CNPAPI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
};

using CUcontext = void*;

struct CUpti_ResourceData {
  CUcontext context;
};

constexpr int CNPAPI_API_ENTER = 0;
constexpr int CNPAPI_API_EXIT = 0;

struct CUpti_CallbackData {
  CUcontext context;
  const char* symbolName;
  int callbackSite;
};
#endif // HAS_CNPAPI
