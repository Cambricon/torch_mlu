/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cnpapi.h>
#include <cn_api.h>

namespace libkineto_mlu {

const char* memcpyKindString(cnpapiActivityMemcpyType kind);
const char* kernelTypeString(uint64_t kernel_type);
const char* runtimeCbidName(cnpapiActivityType type, cnpapi_CallbackId cbid);
const char* overheadKindString(cnpapiActivityOverheadType kind);
const char* atomicOpTypeString(cnpapiActivityAtomicOpType type);
const char* atomicCompareFlagString(cnpapiActivityAtomicCompareFlag flag);
const char* atomicRequestFlagString(cnpapiActivityAtomicRequestFlag flag);

} // namespace libkineto_mlu
