/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cnpapi.h>
#include <chrono>
#include <fmt/format.h>
#include <ITraceActivity.h>
#include <GenericTraceActivity.h>
#include <ThreadUtil.h>
#include <time_since_epoch.h>
#include <output_base.h>

#include "cnpapi_strings.h"
#include "CnpapiRecord.h"

namespace libkineto_mlu {
using namespace libkineto;

GenericTraceActivity createRuntimeActivity(
      const RuntimeRecord* activity,
      const ITraceActivity* linked,
      int32_t threadId);

GenericTraceActivity createOverheadActivity(
      const OverheadRecord* activity,
      const ITraceActivity* linked,
      int32_t threadId=0);

template<class T>
GenericTraceActivity createMluActivity(
      const T* activity, const ITraceActivity* linked);

} // namespace libkineto_mlu
