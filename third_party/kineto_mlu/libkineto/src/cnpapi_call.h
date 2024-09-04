// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fmt/format.h>

#ifdef HAS_CNPAPI

#include <cnpapi.h>

#include "Logger.h"

#define CNPAPI_CALL(call)                           \
  [&]() -> cnpapiResult {                           \
    cnpapiResult _status_ = call;                   \
    if (_status_ != CNPAPI_SUCCESS) {               \
      const char* _errstr_ = nullptr;              \
      cnpapiGetResultString(_status_, &_errstr_);   \
      LOG(ERROR) << fmt::format(                 \
          "function {} failed with error {} ({})", \
          #call,                                   \
          _errstr_,                                \
          (int)_status_);                          \
    }                                              \
    return _status_;                               \
  }()

#define CNPAPI_CALL_NOWARN(call) call

#else

#define CNPAPI_CALL(call) call
#define CNPAPI_CALL_NOWARN(call) call

#endif // HAS_CNPAPI
