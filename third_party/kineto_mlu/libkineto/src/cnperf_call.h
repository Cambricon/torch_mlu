#pragma once

#include <fmt/format.h>

#include <cnperf_api.h>

#include "Logger.h"

using namespace libkineto;

#define CNPERF_CALL(call)                                 \
  [&]() -> cnperfResult_t {                               \
    cnperfResult_t _status_ = call;                       \
    if (_status_ != CNPERF_SUCCESS) {                     \
      LOG(ERROR) << fmt::format(                          \
          "Call function {} failed with error code ({})", \
          #call,                                          \
          (int)_status_);                                 \
    }                                                     \
    return _status_;                                      \
  }()

#define CNPERF_CALL_NOWARN(call) call
