// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "libkineto.h"

#include "Logger.h"

using namespace KINETO_NAMESPACE;
extern "C" {

void suppressLibkinetoLogMessages() {
  SET_LOG_SEVERITY_LEVEL(ERROR);
}

} // extern C
