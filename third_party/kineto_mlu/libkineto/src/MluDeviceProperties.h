/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <string>

#include <cnpapi.h>

namespace libkineto_mlu {

// Return compute properties for each device as a json string
const std::string& devicePropertiesJson();

} // namespace libkineto_mlu
