/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace libkineto_mlu {

std::string demangle(const char* name);
std::string demangle(const std::string& name);

} // namespace libkineto_mlu
