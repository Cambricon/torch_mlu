/*
 * Copyright (c) Cambricon Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cnpapi_strings.h"
#include "cnpapi_call.h"

namespace libkineto_mlu {

const char* memcpyKindString(
    cnpapiActivityMemcpyType kind) {
  switch (kind) {
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_HTOD:
      return "HtoD";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_DTOH:
      return "DtoH";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_DTOD:
      return "DtoD";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_HTOH:
      return "HtoH";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

const char* kernelTypeString(uint64_t kernel_type) {
  switch (kernel_type) {
    case CN_KERNEL_CLASS_BLOCK:
      return "BLOCK";
    case CN_KERNEL_CLASS_UNION:
      return "UNION";
    case CN_KERNEL_CLASS_UNION2:
      return "UNION2";
    case CN_KERNEL_CLASS_UNION4:
      return "UNION4";
    case CN_KERNEL_CLASS_UNION8:
      return "UNION8";
    case CN_KERNEL_CLASS_UNION16:
      return "UNION16";
    default:
      break;
  }
  return "<unknown>";
}

const char* overheadKindString(
    cnpapiActivityOverheadType kind) {
  switch (kind) {
    case CNPAPI_ACTIVITY_OVERHEAD_UNKNOWN:
      return "Unknown";
    case CNPAPI_ACTIVITY_OVERHEAD_CNPAPI_BUFFER_FLUSH:
      return "Buffer Flush";
    case CNPAPI_ACTIVITY_OVERHEAD_CNPAPI_RESOURCE:
      return "Resource";
    default:
      return "Unrecognized";
  }
}

namespace {

cnpapi_CallbackDomain activityType2CallbakDomain(cnpapiActivityType type) {
  cnpapi_CallbackDomain domain = CNPAPI_CB_DOMAIN_FORCE_INT;
  switch (type)
  {
  case CNPAPI_ACTIVITY_TYPE_CNDRV_API:
    domain = CNPAPI_CB_DOMAIN_CNDRV_API;
    break;
  case CNPAPI_ACTIVITY_TYPE_CNRT_API:
    domain = CNPAPI_CB_DOMAIN_CNRT_API;
    break;
  case CNPAPI_ACTIVITY_TYPE_CNNL_API:
    domain = CNPAPI_CB_DOMAIN_CNNL_API;
    break;
  case CNPAPI_ACTIVITY_TYPE_CNCL_API:
    domain = CNPAPI_CB_DOMAIN_CNCL_API;
    break;
  case CNPAPI_ACTIVITY_TYPE_CNNL_EXTRA_API:
    domain = CNPAPI_CB_DOMAIN_CNNL_EXTRA_API;
    break;
  default:
    break;
  }
  return domain;
}

}

const char* runtimeCbidName(cnpapiActivityType type, cnpapi_CallbackId cbid) {
  const char* name;
  CNPAPI_CALL(cnpapiGetCallbackName(activityType2CallbakDomain(type), cbid, &name));
  return name;
}

const char* atomicCompareFlagString(cnpapiActivityAtomicCompareFlag flag) {
  switch (flag) {
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_COMPARE_EQUAL:
      return "EQUAL";
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_COMPARE_LESS_EQUAL:
      return "LESS_EQUAL";
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_COMPARE_LESS:
      return "LESS";
    default:
      break;
  }
  return "<unknown>";
}

const char* atomicRequestFlagString(cnpapiActivityAtomicRequestFlag flag) {
  switch (flag) {
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_REQUEST_TYPE_UNKNOWN:
      return "<unknown>";
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_REQUEST_DEFAULT:
      return "DEFAULT";
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_ADD:
      return "ADD";
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_SET:
      return "SET";
    case CNPAPI_ACTIVITY_FLAG_ATOMIC_CLEAR:
      return "CLEAR";
    default:
      break;
  }
  return "<unknown>";
}

const char* atomicOpTypeString(cnpapiActivityAtomicOpType type) {
  switch (type) {
    case CNPAPI_ACTIVITY_ATOMIC_OP_REQUEST:
      return "REQUEST";
    case CNPAPI_ACTIVITY_ATOMIC_OP_COMPARE:
      return "COMPARE";
    default:
      break;
  }
  return "<unknown>";
}

} // namespace libkineto_mlu
