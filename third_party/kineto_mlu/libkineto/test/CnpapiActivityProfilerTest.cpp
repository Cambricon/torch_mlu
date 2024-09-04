// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <strings.h>
#include <time.h>
#include <chrono>

#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#endif

#include "include/libkineto.h"
#include "include/Config.h"
#include "include/TraceSpan.h"
#include "src/CnpapiActivityProfiler.h"
#include "src/ActivityTrace.h"
#include "src/CnpapiActivityApi.h"
#include "src/output_base.h"
#include "src/output_json.h"
#include "src/output_membuf.h"

#include "src/Logger.h"
#include "test/MockActivitySubProfiler.h"

using namespace std::chrono;
using namespace libkineto;

#define MLU_LAUNCH_KERNEL CNPAPI_CNDRV_TRACE_CBID_cnInvokeKernel
#define MLU_MEMCPY CNPAPI_CNDRV_TRACE_CBID_cnMemcpyAsync

namespace {
const TraceSpan& defaultTraceSpan() {
  static TraceSpan span(0, 0, "Unknown", "");
  return span;
}

static uint64_t mlu_time_point = cnpapiGetTimestamp();
static uint64_t cpu_time_point = libkineto::timeSinceEpoch(std::chrono::system_clock::now());

}

// Provides ability to easily create a few test CPU-side ops
struct MockCpuActivityBuffer : public CpuTraceBuffer {
  MockCpuActivityBuffer(int64_t startTime, int64_t endTime) {
    span = TraceSpan(startTime, endTime, "Test trace");
    mluOpCount = 0;
  }

  void addOp(std::string name, int64_t startTime, int64_t endTime, int64_t correlation) {
    GenericTraceActivity op(span, ActivityType::CPU_OP, name);
    op.startTime = startTime + cpu_time_point;
    op.endTime = endTime + cpu_time_point;
    op.resource = systemThreadId();
    op.id = correlation;
    emplace_activity(std::move(op));
    span.opCount++;
  }
};

// Provides ability to easily create a few test mlu ops
struct MockCnpapiActivityBuffer {
  void addCorrelationActivity(int64_t correlation, cnpapiActivityExternalCorrelationType externalKind, int64_t externalId) {
    auto& act = *(cnpapiActivityExternalCorrelation*) malloc(sizeof(cnpapiActivityExternalCorrelation));
    act.type = CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION;
    act.external_id = externalId;
    act.external_type = externalKind;
    act.correlation_id = correlation;
    activities.push_back(reinterpret_cast<cnpapiActivity*>(&act));
  }

  void addRuntimeActivity(
      cnpapiActivityType type,
      cnpapi_CallbackId cbid,
      int64_t start_us, int64_t end_us, int64_t correlation) {
    auto& act = createActivity<cnpapiActivityAPI>(
        start_us, end_us, correlation);
    act.type = type;
    act.cbid = cbid;
    act.thread_id = threadId();
    activities.push_back(reinterpret_cast<cnpapiActivity*>(&act));
  }

  void addKernelActivity(
      int64_t start_us, int64_t end_us, int64_t correlation) {
    auto& act = createActivity<cnpapiActivityKernel>(
        start_us, end_us, correlation);
    act.type = CNPAPI_ACTIVITY_TYPE_KERNEL;
    act.device_id = 0;
    act.queue_id = 1;
    act.name = "kernel";
    act.dimx = act.dimy = act.dimz = 1;
    activities.push_back(reinterpret_cast<cnpapiActivity*>(&act));
  }

  void addMemcpyActivity(
      int64_t start_us, int64_t end_us, int64_t correlation) {
    auto& act = createActivity<cnpapiActivityMemcpy>(
        start_us, end_us, correlation);
    act.type = CNPAPI_ACTIVITY_TYPE_MEMCPY;
    act.device_id = 0;
    act.queue_id = 2;
    act.copy_type = CNPAPI_ACTIVITY_MEMCPY_TYPE_HTOD;
    activities.push_back(reinterpret_cast<cnpapiActivity*>(&act));
  }

  template<class T>
  T& createActivity(
      int64_t start_us, int64_t end_us, int64_t correlation) {
    T& act = *static_cast<T*>(malloc(sizeof(T)));
    bzero(&act, sizeof(act));
    act.start = start_us * 1000 + mlu_time_point;
    act.end = end_us * 1000 + mlu_time_point;
    act.correlation_id = correlation;
    return act;
  }

  ~MockCnpapiActivityBuffer() {
    for (cnpapiActivity* act : activities) {
      free(act);
    }
  }

  std::vector<cnpapiActivity*> activities;
};

// Mock parts of the CnpapiActivityApi
class MockCnpapiActivities : public CnpapiActivityApi {
 public:
  void bufferRequestedOverride(uint64_t** buffer, size_t* size, size_t* maxNumRecords) {
    this->bufferRequested(buffer, size, maxNumRecords);
  }

  void bufferCompletedOverride() {
    for (auto& act : activityBuffer->activities) {
      this->processRecords(act);
    }
  }

  std::unique_ptr<MockCnpapiActivityBuffer> activityBuffer;
};


// Common setup / teardown and helper functions
class CnpapiActivityProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cfg_ = std::make_unique<KINETO_NAMESPACE::Config>();
    cfg_->validate(std::chrono::system_clock::now());
    std::set<ActivityType> default_act_types;
    for(auto type: defaultActivityTypes()) {
      default_act_types.insert(type);
    }
    cfg_->setSelectedActivityTypes(default_act_types);
    loggerFactory.addProtocol("file", [](const std::string& url) {
        return std::unique_ptr<ActivityLogger>(new ChromeTraceLogger(url));
    });
  }

  std::unique_ptr<KINETO_NAMESPACE::Config> cfg_;
  MockCnpapiActivities cnpapiActivities_;
  ActivityLoggerFactory loggerFactory;
};

void checkTracefile(const char* filename) {
#ifdef __linux__
  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
  close(fd);
#endif
}

TEST(CnpapiActivityProfiler, AsyncTrace) {
  std::vector<std::string> log_modules(
      {"CnpapiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  MockCnpapiActivities activities;
  CnpapiActivityProfiler profiler(activities, /*cpu only*/ true);

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  EXPECT_TRUE(mkstemps(filename, 5));

  KINETO_NAMESPACE::Config cfg;

  int iter = 0;
  int warmup = 5;
  auto now = system_clock::now();
  auto startTime = now + seconds(10);

  bool success = cfg.parse(fmt::format(R"CFG(
    ACTIVITIES_WARMUP_PERIOD_SECS = {}
    ACTIVITIES_DURATION_SECS = 1
    ACTIVITIES_LOG_FILE = {}
    PROFILE_START_TIME = {}
  )CFG", warmup, filename, duration_cast<milliseconds>(startTime.time_since_epoch()).count()));

  EXPECT_TRUE(success);
  EXPECT_FALSE(profiler.isActive());

  auto logger = std::make_unique<ChromeTraceLogger>(cfg.activitiesLogFile());

  std::set<ActivityType> default_act_types;
  for(auto type: defaultActivityTypes()) {
    default_act_types.insert(type);
  }
  cfg.setSelectedActivityTypes(default_act_types);

  // Usually configuration is done when now is startTime - warmup to kick off warmup
  // but start right away in the test
  profiler.configure(cfg, now);
  profiler.setLogger(logger.get());

  EXPECT_TRUE(profiler.isActive());

  // fast forward in time and we have reached the startTime
  now = startTime;

  // Run the profiler
  // Warmup
  // performRunLoopStep is usually called by the controller loop and takes
  // the current time and the controller's next wakeup time.
  profiler.performRunLoopStep(
      /* Current time */ now, /* Next wakeup time */ now);

  auto next = now + milliseconds(1000);

  // performRunLoopStep can also be called by an application thread to update iteration count
  // since this config does not use iteration this should have no effect on the state
  while (++iter < 20) {
    profiler.performRunLoopStep(now, now, iter);
  }

  // Runloop should now be in collect state, so start workload
  // Perform another runloop step, passing in the end profile time as current.
  // This should terminate collection
  profiler.performRunLoopStep(
      /* Current time */ next, /* Next wakeup time */ next);
  // One step needed for each of the Process and Finalize phases
  // Doesn't really matter what times we pass in here.

  EXPECT_TRUE(profiler.isActive());

  auto nextnext = next + milliseconds(1000);

  while (++iter < 40) {
    profiler.performRunLoopStep(next, next, iter);
  }

  EXPECT_TRUE(profiler.isActive());

  profiler.performRunLoopStep(nextnext,nextnext);
  profiler.performRunLoopStep(nextnext,nextnext);

  // Assert that tracing has completed
  EXPECT_FALSE(profiler.isActive());

  checkTracefile(filename);
}

TEST(CnpapiActivityProfiler, AsyncTraceUsingIter) {
  std::vector<std::string> log_modules(
      {"CnpapiActivityProfiler.cpp", "output_json.cpp"});
  SET_LOG_VERBOSITY_LEVEL(1, log_modules);

  auto runIterTest = [&](
    int start_iter, int warmup_iters, int trace_iters) {

    LOG(INFO ) << "Async Trace Test: start_iteration = " << start_iter
               << " warmup iterations = " << warmup_iters
               << " trace iterations = " << trace_iters;

    MockCnpapiActivities activities;
    CnpapiActivityProfiler profiler(activities, /*cpu only*/ true);

    char filename[] = "/tmp/libkineto_testXXXXXX.json";
    EXPECT_TRUE(mkstemps(filename, 5));

    KINETO_NAMESPACE::Config cfg;

    int iter = 0;
    auto now = system_clock::now();

    bool success = cfg.parse(fmt::format(R"CFG(
      PROFILE_START_ITERATION = {}
      ACTIVITIES_WARMUP_ITERATIONS={}
      ACTIVITIES_ITERATIONS={}
      ACTIVITIES_DURATION_SECS = 1
      ACTIVITIES_LOG_FILE = {}
    )CFG", start_iter, warmup_iters, trace_iters, filename));

    EXPECT_TRUE(success);
    EXPECT_FALSE(profiler.isActive());

    auto logger = std::make_unique<ChromeTraceLogger>(cfg.activitiesLogFile());

    // Usually configuration is done when now is startIter - warmup iter to kick off warmup
    // but start right away in the test
    while (iter < (start_iter - warmup_iters)) {
      profiler.performRunLoopStep(now, now, iter++);
    }

    std::set<ActivityType> default_act_types;
    for(auto type: defaultActivityTypes()) {
      default_act_types.insert(type);
    }
    cfg.setSelectedActivityTypes(default_act_types);

    profiler.configure(cfg, now);
    profiler.setLogger(logger.get());

    EXPECT_TRUE(profiler.isActive());

    // fast forward in time, mimicking what will happen in reality
    now += seconds(10);
    auto next = now + milliseconds(1000);

    // this call to runloop step should not be effecting the state
    profiler.performRunLoopStep(now, next);
    EXPECT_TRUE(profiler.isActive());

    // start trace collection
    while (iter < start_iter) {
      profiler.performRunLoopStep(now, next, iter++);
    }

    // Runloop should now be in collect state, so start workload

    while (iter < (start_iter + trace_iters)) {
      profiler.performRunLoopStep(now, next, iter++);
    }

    // One step is required for each of the Process and Finalize phases
    // Doesn't really matter what times we pass in here.
    if (iter >= (start_iter + trace_iters)) {
      profiler.performRunLoopStep(now, next, iter++);
    }
    EXPECT_TRUE(profiler.isActive());

    auto nextnext = next + milliseconds(1000);

    profiler.performRunLoopStep(nextnext, nextnext);
    profiler.performRunLoopStep(nextnext, nextnext);

    // Assert that tracing has completed
    EXPECT_FALSE(profiler.isActive());

    checkTracefile(filename);
  };

  // start iter = 50, warmup iters = 5, trace iters = 10
  runIterTest(50, 5, 10);
  // should be able to start at 0 iteration
  runIterTest(0, 0, 2);
  runIterTest(0, 5, 5);
}

TEST_F(CnpapiActivityProfilerTest, SyncTrace) {
  using ::testing::Return;
  using ::testing::ByMove;

  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules(
      {"CnpapiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CnpapiActivityProfiler profiler(cnpapiActivities_, /*cpu only*/ false);
  int64_t start_time_us = 100 + cpu_time_point;
  int64_t duration_us = 300;
  auto start_time = time_point<system_clock>(microseconds(start_time_us));
  profiler.configure(*cfg_, start_time);

  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + microseconds(duration_us));

  profiler.recordThreadInfo();

  // Log some cpu ops
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_us, start_time_us + duration_us);
  cpuOps->addOp("op1", 120, 150, 1);
  cpuOps->addOp("op2", 130, 140, 2);
  cpuOps->addOp("op3", 200, 250, 3);
  profiler.transferCpuTrace(std::move(cpuOps));

  // And some MLU ops
  auto mluOps = std::make_unique<MockCnpapiActivityBuffer>();
  mluOps->addRuntimeActivity(CNPAPI_ACTIVITY_TYPE_CNDRV_API, MLU_LAUNCH_KERNEL, 133, 138, 1);
  mluOps->addRuntimeActivity(CNPAPI_ACTIVITY_TYPE_CNDRV_API, MLU_MEMCPY, 210, 220, 2);
  mluOps->addRuntimeActivity(CNPAPI_ACTIVITY_TYPE_CNDRV_API, MLU_LAUNCH_KERNEL, 230, 245, 3);
  mluOps->addKernelActivity(150, 170, 1);
  mluOps->addMemcpyActivity(240, 250, 2);
  mluOps->addKernelActivity(260, 320, 3);
  cnpapiActivities_.activityBuffer = std::move(mluOps);
  cnpapiActivities_.bufferCompletedOverride();

  // Have the profiler process them
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  // Wrapper that allows iterating over the activities
  ActivityTrace trace(std::move(logger), loggerFactory);
  EXPECT_EQ(trace.activities()->size(), 9);
  std::map<std::string, int> activityCounts;
  std::map<int64_t, int> resourceIds;
  for (auto& activity : *trace.activities()) {
    activityCounts[activity->name()]++;
    resourceIds[activity->resourceId()]++;
  }
  for (const auto& p : activityCounts) {
    LOG(INFO) << p.first << ": " << p.second;
  }
  EXPECT_EQ(activityCounts["op1"], 1);
  EXPECT_EQ(activityCounts["op2"], 1);
  EXPECT_EQ(activityCounts["op3"], 1);
  EXPECT_EQ(activityCounts["cnInvokeKernel"], 2);
  EXPECT_EQ(activityCounts["cnMemcpyAsync"], 1);
  EXPECT_EQ(activityCounts["kernel"], 2);
  EXPECT_EQ(activityCounts["Memcpy HtoD"], 1);

  auto sysTid = systemThreadId();
  // Ops and runtime events are on thread sysTid
  EXPECT_EQ(resourceIds[sysTid], 6);
  // Kernels are on stream 1, memcpy on stream 2
  EXPECT_EQ(resourceIds[1], 2);
  EXPECT_EQ(resourceIds[2], 1);

#ifdef __linux__
  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  EXPECT_TRUE(mkstemps(filename, 5));
  trace.save(filename);
  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);
  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
#endif
}

TEST_F(CnpapiActivityProfilerTest, MluUserAnnotationTest) {
  GTEST_SKIP() << "MLU_USER_ANNOTATION is not enabled.";
  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules(
      {"CnpapiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Start and stop profiling
  CnpapiActivityProfiler profiler(cnpapiActivities_, /*cpu only*/ false);
  int64_t start_time_us = 100 + cpu_time_point;
  int64_t duration_us = 300;
  auto start_time = time_point<system_clock>(microseconds(start_time_us));
  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  profiler.stopTrace(start_time + microseconds(duration_us));

  int64_t kernelLaunchTime = 120;
  profiler.recordThreadInfo();

  // set up CPU event
  auto cpuOps = std::make_unique<MockCpuActivityBuffer>(
      start_time_us, start_time_us + duration_us);
  cpuOps->addOp("annotation", kernelLaunchTime, kernelLaunchTime + 10, 1);
  profiler.transferCpuTrace(std::move(cpuOps));

  // set up a couple of MLU events and correlate with above CPU event.
  // CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1 is used for user annotations.
  auto mluOps = std::make_unique<MockCnpapiActivityBuffer>();
  mluOps->addCorrelationActivity(1, CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, 1);
  mluOps->addKernelActivity(kernelLaunchTime + 5, kernelLaunchTime + 10, 1);
  mluOps->addCorrelationActivity(1, CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, 1);
  mluOps->addKernelActivity(kernelLaunchTime + 15, kernelLaunchTime + 25, 1);
  cnpapiActivities_.activityBuffer = std::move(mluOps);
  cnpapiActivities_.bufferCompletedOverride();

  // process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  std::map<std::string, int> counts;
  for (auto& activity : *trace.activities()) {
    LOG(ERROR) << "name: " << activity->name();
    counts[activity->name()]++;
  }

  // We should now have an additional annotation activity created
  // on the MLU timeline.
  EXPECT_EQ(counts["annotation"], 2);
  EXPECT_EQ(counts["kernel"], 2);

  auto& annotation = trace.activities()->at(0);
  auto& kernel1 = trace.activities()->at(1);
  auto& kernel2 = trace.activities()->at(2);
  auto& mlu_annotation = trace.activities()->at(3);

  // TODO(SYSTOOL-3625): kernel order is not sure.
  bool has_order = kernel1->timestamp() < kernel2->timestamp();
  
  auto assert_func = [&](const libkineto::ITraceActivity* const& kernel_prev,
                         const libkineto::ITraceActivity* const& kernel_back) {
    EXPECT_EQ(mlu_annotation->type(), ActivityType::MLU_USER_ANNOTATION);
    EXPECT_EQ(mlu_annotation->timestamp(), kernel_prev->timestamp());
    EXPECT_EQ(
        mlu_annotation->duration(),
        kernel_back->timestamp() + kernel_back->duration() - kernel_prev->timestamp());
    EXPECT_EQ(mlu_annotation->deviceId(), kernel_prev->deviceId());
    EXPECT_EQ(mlu_annotation->resourceId(), kernel_prev->resourceId());
    EXPECT_EQ(mlu_annotation->correlationId(), annotation->correlationId());
    EXPECT_EQ(mlu_annotation->name(),  annotation->name());
  };

  if (has_order) {
    assert_func(kernel1, kernel2);
  } else {
    assert_func(kernel2, kernel1);
  }
}

TEST_F(CnpapiActivityProfilerTest, SubActivityProfilers) {
  using ::testing::Return;
  using ::testing::ByMove;

  // Verbose logging is useful for debugging
  std::vector<std::string> log_modules(
      {"CnpapiActivityProfiler.cpp"});
  SET_LOG_VERBOSITY_LEVEL(2, log_modules);

  // Setup example events to test
  GenericTraceActivity ev{defaultTraceSpan(), ActivityType::GLOW_RUNTIME, ""};
  ev.device = 1;
  ev.resource = 0;

  int64_t start_time_us = 100 + cpu_time_point;
  int64_t duration_us = 1000;
  auto start_time = time_point<system_clock>(microseconds(start_time_us));

  std::deque<GenericTraceActivity> test_activities{3, ev};
  test_activities[0].startTime = start_time_us;
  test_activities[0].endTime = start_time_us + 5000;
  test_activities[0].activityName = "SubGraph A execution";
  test_activities[1].startTime = start_time_us;
  test_activities[1].endTime = start_time_us + 2000;
  test_activities[1].activityName = "Operator foo";
  test_activities[2].startTime = start_time_us + 2500;
  test_activities[2].endTime = start_time_us + 2900;
  test_activities[2].activityName = "Operator bar";

  auto mock_activity_profiler =
    std::make_unique<MockActivityProfiler>(test_activities);

  MockCnpapiActivities activities;
  CnpapiActivityProfiler profiler(activities, /*cpu only*/ true);
  profiler.addChildActivityProfiler(
      std::move(mock_activity_profiler));

  profiler.configure(*cfg_, start_time);
  profiler.startTrace(start_time);
  EXPECT_TRUE(profiler.isActive());

  profiler.stopTrace(start_time + microseconds(duration_us));
  EXPECT_TRUE(profiler.isActive());

  char filename[] = "/tmp/libkineto_testXXXXXX.json";
  EXPECT_TRUE(mkstemps(filename, 5));
  LOG(INFO) << "Logging to tmp file " << filename;

  // process trace
  auto logger = std::make_unique<MemoryTraceLogger>(*cfg_);
  profiler.processTrace(*logger);
  profiler.setLogger(logger.get());

  // Profiler can be reset at this point - logger owns the activities
  profiler.reset();

  ActivityTrace trace(std::move(logger), loggerFactory);
  trace.save(filename);
  const auto& traced_activites = trace.activities();

  // Test we have all the events
  EXPECT_EQ(traced_activites->size(), test_activities.size());

  // Check that the expected file was written and that it has some content
  int fd = open(filename, O_RDONLY);
  if (!fd) {
    perror(filename);
  }
  EXPECT_TRUE(fd);

  // Should expect at least 100 bytes
  struct stat buf{};
  fstat(fd, &buf);
  EXPECT_GT(buf.st_size, 100);
}

TEST_F(CnpapiActivityProfilerTest, BufferSizeLimitTestWarmup) {
  CnpapiActivityProfiler profiler(cnpapiActivities_, /*cpu only*/ false);

  auto now = system_clock::now();
  auto startTime = now + seconds(10);

  int maxBufferSizeMB = 3;

  auto startTimeEpoch = std::to_string(duration_cast<milliseconds>(startTime.time_since_epoch()).count());
  std::string maxBufferSizeMBStr = std::to_string(maxBufferSizeMB);
  cfg_->handleOption("ACTIVITIES_MAX_MLU_BUFFER_SIZE_MB", maxBufferSizeMBStr);
  cfg_->handleOption("PROFILE_START_TIME", startTimeEpoch);


  EXPECT_FALSE(profiler.isActive());
  profiler.configure(*cfg_, now);
  EXPECT_TRUE(profiler.isActive());

  for (size_t i = 0; i < maxBufferSizeMB; i++) {
    uint64_t* buf;
    size_t mluBufferSize;
    size_t maxNumRecords;
    cnpapiActivities_.bufferRequestedOverride(&buf, &mluBufferSize, &maxNumRecords);
  }

  // fast forward to startTime and profiler is now running
  now = startTime;

  profiler.performRunLoopStep(now, now);

  auto next = now + milliseconds(1000);
  profiler.performRunLoopStep(next, next);
  profiler.performRunLoopStep(next, next);
  profiler.performRunLoopStep(next, next);

  EXPECT_FALSE(profiler.isActive());
}
