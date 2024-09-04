#include <cnrt.h>
#include <cnpapi_generated_cnrt_params.h>
#include <gtest/gtest.h>

#include "include/CnpapiCallbackManager.h"
#include "src/CnpapiActivityApi.h"

using namespace libkineto_mlu;

class CustomCallback {
  public:
    static CustomCallback& singleton() {
        static CustomCallback singleton;
        return singleton;
    }

    void payload() {
        // Prepare input and output
        char *h0 = NULL;
        char *h1 = NULL;
        cnrtHostMalloc((void **)&h0, malloc_size);
        cnrtHostMalloc((void **)&h1, malloc_size);
        memset(h0, 'a', malloc_size);

        void *d = NULL;
        cnrtMalloc((void **)&d, malloc_size);

        // Create queue
        cnrtQueue_t queue;
        cnrtQueueCreate(&queue);

        // Memcpy Async
        cnrtMemcpyAsync(d, h0, malloc_size, queue, cnrtMemcpyHostToDev);
        cnrtMemcpyAsync(h1, d, malloc_size, queue, cnrtMemcpyDevToHost);
        cnrtQueueSync(queue);

        // Free resource
        cnrtQueueDestroy(queue);
        cnrtFreeHost(h0);
        cnrtFreeHost(h1);
        cnrtFree(d);
    }

    static void callback_function(
        void* userdata,
        cnpapi_CallbackDomain domain,
        cnpapi_CallbackId cbid,
        const cnpapi_CallbackData* cbdata) {
      if (domain == CNPAPI_CB_DOMAIN_CNRT_API && cbdata->callbackSite == CNPAPI_API_EXIT) { 
        if (cbid == CNPAPI_CNRT_TRACE_CBID_cnrtMemcpyAsync) {
          singleton().found_cnrt_memcpy_count += 1;
          singleton().found_cnrt_memcpy_size +=
            reinterpret_cast<const cnpapi_cnrtMemcpyAsync_params*>(cbdata->functionParams)->bytes;
        }
        if (cbid == CNPAPI_CNRT_TRACE_CBID_cnrtMalloc) {
          singleton().found_cnrt_malloc_count += 1;
        }
        if (cbid == CNPAPI_CNRT_TRACE_CBID_cnrtFreeHost) {
          singleton().found_cnrt_free_host_count += 1;
        }
        if (cbid == CNPAPI_CNRT_TRACE_CBID_cnrtFree) {
          singleton().found_cnrt_free_count += 1;
        }
      }
   }

    int found_cnrt_malloc_count = 0;
    int found_cnrt_memcpy_count = 0;
    int found_cnrt_free_host_count = 0;
    int found_cnrt_free_count = 0;
    int found_cnrt_memcpy_size = 0;
    const size_t malloc_size = sizeof(size_t) * 10;
};

TEST(CnpapiCallbackManagerTest, registerCallbackTest) {
  // Register custom callback
  kineto_mlu::CnpapiCallbackManager::getInstance().registerCallbackFunction(
      (cnpapi_CallbackFunc)CustomCallback::callback_function);
  kineto_mlu::CnpapiCallbackManager::getInstance().insertEnabledCbid(
      CNPAPI_CB_DOMAIN_CNRT_API, CNPAPI_CNRT_TRACE_CBID_cnrtFreeHost);

  // Start profiler
  constexpr size_t maxBufSize(20 * 1024 * 1024);  // 20MB
  CnpapiActivityApi::singleton().setMaxBufferSize(maxBufSize);
  CnpapiActivityApi::singleton().enableCnpapiActivities({ActivityType::MLU_RUNTIME});
  CustomCallback::singleton().payload();
  CnpapiActivityApi::singleton().disableCnpapiActivities({ActivityType::MLU_RUNTIME});

  // Enabled by default. Called 1 time
  EXPECT_EQ(CustomCallback::singleton().found_cnrt_malloc_count, 1);
  // Enabled by default. Called 2 times
  EXPECT_EQ(CustomCallback::singleton().found_cnrt_memcpy_count, 2);
  // Disabled by default but enabled using
  // kineto_mlu::CnpapiCallbackManager::getInstance().insertEnabledCbid.
  // Called 2 times
  EXPECT_EQ(CustomCallback::singleton().found_cnrt_free_host_count, 2);
  // Disabled by default
  EXPECT_EQ(CustomCallback::singleton().found_cnrt_free_count, 0);
  // size = H2D + D2H = malloc_size * 2
  EXPECT_EQ(CustomCallback::singleton().found_cnrt_memcpy_size,
            CustomCallback::singleton().malloc_size * 2);
}
