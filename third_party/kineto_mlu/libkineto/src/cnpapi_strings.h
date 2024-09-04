// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cnpapi.h>
#include <cn_api.h>

namespace libkineto {

const char* memcpyKindString(cnpapiActivityMemcpyType kind);
const char* kernelTypeString(uint64_t kernel_type);
const char* runtimeCbidName(cnpapiActivityType type, cnpapi_CallbackId cbid);
const char* overheadKindString(cnpapiActivityOverheadType kind);
const char* atomicOpTypeString(cnpapiActivityAtomicOpType type);
const char* atomicCompareFlagString(cnpapiActivityAtomicCompareFlag flag);
const char* atomicRequestFlagString(cnpapiActivityAtomicRequestFlag flag);

} // namespace libkineto
