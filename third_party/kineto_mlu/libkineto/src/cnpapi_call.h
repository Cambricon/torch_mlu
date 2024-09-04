/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
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

