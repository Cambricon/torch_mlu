// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "src/cnpapi_strings.h"

using namespace libkineto_mlu;

TEST(CnpapiStringsTest, kernelTypeStringTest) {
  ASSERT_EQ(cnpapiInit(), CNPAPI_SUCCESS);

  ASSERT_STREQ(kernelTypeString(CN_KERNEL_CLASS_BLOCK), "BLOCK");
  ASSERT_STREQ(kernelTypeString(CN_KERNEL_CLASS_UNION), "UNION");
  ASSERT_STREQ(kernelTypeString(CN_KERNEL_CLASS_UNION8), "UNION8");
}

TEST(CnpapiStringsTest, runtimeCbidNameTest) {
  ASSERT_EQ(cnpapiInit(), CNPAPI_SUCCESS);

  ASSERT_STREQ(
      runtimeCbidName(CNPAPI_ACTIVITY_TYPE_CNDRV_API, CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel), "cnInvokeKernel");
  ASSERT_STREQ(
      runtimeCbidName(CNPAPI_ACTIVITY_TYPE_CNDRV_API, CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync), "cnMemcpyAsync");
  ASSERT_STREQ(
      runtimeCbidName(CNPAPI_ACTIVITY_TYPE_CNRT_API, CNPAPI_CNRT_TRACE_CBID_cnrtSyncDevice), "cnrtSyncDevice");
  ASSERT_STREQ(
      runtimeCbidName(CNPAPI_ACTIVITY_TYPE_CNNL_API, CNPAPI_CNNL_TRACE_CBID_cnnlOpTensor), "cnnlOpTensor");
  ASSERT_STREQ(
      runtimeCbidName(CNPAPI_ACTIVITY_TYPE_CNCL_API, CNPAPI_CNCL_TRACE_CBID_cnclAllReduce), "cnclAllReduce");
}

TEST(CnpapiStringsTest, memcpyKindStringTest) {
  ASSERT_EQ(cnpapiInit(), CNPAPI_SUCCESS);

  ASSERT_STREQ(memcpyKindString(CNPAPI_ACTIVITY_MEMCPY_TYPE_HTOD), "HtoD");
  ASSERT_STREQ(memcpyKindString(CNPAPI_ACTIVITY_MEMCPY_TYPE_DTOH), "DtoH");
  ASSERT_STREQ(memcpyKindString(CNPAPI_ACTIVITY_MEMCPY_TYPE_PTOP), "PtoP");
}
