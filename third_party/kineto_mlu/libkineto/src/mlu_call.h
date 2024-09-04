// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fmt/format.h>

#ifdef HAS_CNPAPI

#include <cnrt.h>

#define CNRT_CALL(call)                                      \
  [&]() -> cnrtRet_t {                                     \
    cnrtRet_t _status_ = call;                             \
    if (_status_ != cnrtSuccess) {                           \
      const char* _errstr_ = cnrtGetErrorStr(_status_);   \
      LOG(WARNING) << fmt::format(                           \
          "function {} failed with error {} ({})",           \
          #call,                                             \
          _errstr_,                                          \
          (int)_status_);                                    \
    }                                                        \
    return _status_;                                         \
  }()

#endif // HAS_CNPAPI
