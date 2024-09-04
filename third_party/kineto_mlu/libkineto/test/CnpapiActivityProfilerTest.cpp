#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <output_base.h>
#include <time_since_epoch.h>
#include <strings.h>
#include <cnrt.h>

#include "src/CnpapiActivityProfiler.h"
#include "MockTraceLogger.h"

using namespace libkineto;
using namespace libkineto_mlu;

const int64_t EXTERNAL_ID = 12345;
const ITraceActivity* getLinkedAct(int32_t) {
  static GenericTraceActivity generic_act = GenericTraceActivity();
  generic_act.activityName = "MockCPU_OP";
  generic_act.activityType = ActivityType::CPU_OP;
  generic_act.id = EXTERNAL_ID;
  return &generic_act;
}

const int N = 10;
int payload() {
  // Prepare input and output
  size_t size = sizeof(size_t) * N;
  char *h0 = NULL;
  char *h1 = NULL;
  cnrtHostMalloc((void **)&h0, size);
  cnrtHostMalloc((void **)&h1, size);
  memset(h0, 'a', size);

  void *d = NULL;
  cnrtMalloc((void **)&d, size);

  // Create queue
  cnrtQueue_t queue;
  cnrtQueueCreate(&queue);

  // Memcpy Async
  cnrtMemcpyAsync(d, h0, size, queue, cnrtMemcpyHostToDev);
  cnrtMemcpyAsync(h1, d, size, queue, cnrtMemcpyDevToHost);
  cnrtQueueSync(queue);

  // Free resource
  cnrtQueueDestroy(queue);
  cnrtFreeHost(h0);
  cnrtFreeHost(h1);
  cnrtFree(d);

  return 0;
}

// Common setup / teardown and helper functions
class CnpapiActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for(auto type: defaultActivityTypes()) {
      default_act_types.insert(type);
      privateuse1_act_types.insert(type);
    }
    privateuse1_act_types.insert(ActivityType::PRIVATEUSE1_RUNTIME);
    
    profiler_ = std::make_unique<CnpapiActivityProfiler>();
  }

  std::unique_ptr<CnpapiActivityProfiler> profiler_;
  std::set<ActivityType> default_act_types;
  std::set<ActivityType> privateuse1_act_types;
};

TEST_F(CnpapiActivityProfilerTest, PrivateUse1SessionCreateTest) {
  Config cfg;
  cfg.setSelectedActivityTypes(privateuse1_act_types);

  auto session1 = profiler_->configure(privateuse1_act_types /*not used*/, cfg);
  EXPECT_NE(session1, nullptr);

  auto session2 = profiler_->configure(0, 0, privateuse1_act_types, cfg);
  EXPECT_NE(session2, nullptr);
}

TEST_F(CnpapiActivityProfilerTest, ProfilerNameTest) {
  EXPECT_EQ(profiler_->name(), "CnpapiActivityProfiler");
}

TEST_F(CnpapiActivityProfilerTest, AvailableActivitiesTest) {
  auto availabel_acts = profiler_->availableActivities();
  const std::set<ActivityType> expected_activities =
                                {ActivityType::GPU_USER_ANNOTATION,
                                 ActivityType::GPU_MEMCPY,
                                 ActivityType::GPU_MEMSET,
                                 ActivityType::CONCURRENT_KERNEL,
                                 ActivityType::EXTERNAL_CORRELATION,
                                 ActivityType::PRIVATEUSE1_RUNTIME,
                                 ActivityType::OVERHEAD};
  EXPECT_EQ(availabel_acts, expected_activities);
}

TEST_F(CnpapiActivityProfilerTest, CnpapiActivityProcessingTest) {
  int64_t start_time = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  Config cfg;
  cfg.setSelectedActivityTypes(privateuse1_act_types);
  // Will create and enable cnpapi
  auto session = profiler_->configure(privateuse1_act_types /*not used*/, cfg);
  session->pushCorrelationId(EXTERNAL_ID);

  payload();

  session->popCorrelationId();
  session->stop();
  int64_t end_time = libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  auto logger = std::make_unique<MockTraceLogger>();
  session->processTrace(
    *logger,
    std::bind(getLinkedAct, std::placeholders::_1),
    start_time, end_time);
  auto trace_buffer = session->getTraceBuffer();

  // memcpy D2H kernel and memcpy H2D kernel
  EXPECT_EQ(trace_buffer->span.opCount, 2);
  EXPECT_EQ(trace_buffer->activities.size(), logger->getActivities().size());
  
  // check runtime apis and kernels could be catched
  bool found_cnrtMemcpyAsync = false;
  bool found_cnMemcpyHtoDAsync = false;
  bool found_cnMemcpyDtoHAsync = false;
  bool found_memcpy_h2d_kernel = false;
  bool found_memcpy_d2h_kernel = false;
  bool found_cnrtQueueSync = false;
  std::string memcpy_bytes_h2d = "";
  std::string memcpy_bytes_d2h = "";
  int device_id = -1;
  std::set<int> resource_ids;
  const std::string expected_bytes_str = std::to_string(sizeof(size_t) * N);
  for (const auto& act : trace_buffer->activities) {
    if (act->activityType == ActivityType::PRIVATEUSE1_RUNTIME) {
      EXPECT_EQ(act->linked->correlationId(), EXTERNAL_ID);
      if (act->activityName == "cnrtMemcpyAsync") {
        found_cnrtMemcpyAsync = true;
      }
      if (act->activityName == "cnMemcpyHtoDAsync") {
        found_cnMemcpyHtoDAsync = true;
      }
      if (act->activityName == "cnMemcpyDtoHAsync") {
        found_cnMemcpyDtoHAsync = true;
      }
      if (act->activityName == "cnrtQueueSync") {
        found_cnrtQueueSync = true;
      }
    } else if (act->activityType == ActivityType::GPU_MEMCPY) {
      EXPECT_EQ(act->linked->correlationId(), EXTERNAL_ID);
      if (device_id == -1) {
        // get the first kernel devcie id
        device_id = act->device;
      } else {
        // assert all kernels have same devcie id
        EXPECT_EQ(act->device, device_id);
      }
      resource_ids.insert(act->resource);
      if (act->activityName == "Memcpy HtoD") {
        found_memcpy_h2d_kernel = true;
        memcpy_bytes_h2d = act->getMetadataValue("bytes");
      }
      if (act->activityName == "Memcpy DtoH") {
        found_memcpy_d2h_kernel = true;
        memcpy_bytes_d2h = act->getMetadataValue("bytes");
      }
    }
  }
  EXPECT_TRUE(found_cnrtMemcpyAsync);
  EXPECT_TRUE(found_cnMemcpyHtoDAsync);
  EXPECT_TRUE(found_cnMemcpyDtoHAsync);
  EXPECT_TRUE(found_cnrtQueueSync);
  EXPECT_TRUE(found_memcpy_h2d_kernel);
  EXPECT_TRUE(found_memcpy_d2h_kernel);
  EXPECT_EQ(memcpy_bytes_h2d, expected_bytes_str);
  EXPECT_EQ(memcpy_bytes_d2h, expected_bytes_str);

  auto device_info = session->getDeviceInfo();
  EXPECT_EQ(device_info->id, device_id);
  EXPECT_EQ(device_info->label, "MLU");

  auto resource_infos = session->getResourceInfos();
  for (auto res_info : resource_infos) {
    EXPECT_EQ(res_info.deviceId, device_id);
    EXPECT_NE(resource_ids.find(res_info.id), resource_ids.end());
  }
}
