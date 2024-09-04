#include <gtest/gtest.h>
#include <filesystem>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include "framework/core/device.h"
#include "framework/distributed/process_group_cncl.hpp"
#include "framework/distributed/cncl_utils.h"

namespace torch_mlu {

// This is copied from test/cpp/c10d/TestUtils.hpp of pytorch
std::string tmppath() {
  // TMPFILE is for manual test execution during which the user will specify
  // the full temp file path using the environmental variable TMPFILE
  const char* tmpfile = getenv("TMPFILE");
  if (tmpfile) {
    return std::string(tmpfile);
  }

  const char* tmpdir = getenv("TMPDIR");
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";
  }

  // Create template
  std::vector<char> tmp(256);
  auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);
  tmp.resize(len);

  // Create temporary file
  auto fd = mkstemp(&tmp[0]);
  if (fd == -1) {
    throw std::system_error(errno, std::system_category());
  }
  close(fd);
  return std::string(tmp.data(), tmp.size());
}

struct TemporaryFile {
  std::string path;

  TemporaryFile() {
    path = tmppath();
  }

  ~TemporaryFile() {
    unlink(path.c_str());
  }
};

class WorkCNCLSimulateErrors : public ProcessGroupCNCL::WorkCNCL {
 public:
  WorkCNCLSimulateErrors(
      at::Device& device,
      bool simulate_error,
      int rank,
      c10d::OpType opType,
      uint64_t seq)
      : WorkCNCL(device, rank, opType, seq), simulateError_(simulate_error) {}

  std::exception_ptr checkForCNCLErrors() override {
    if (simulateError_) {
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    return ProcessGroupCNCL::WorkCNCL::checkForCNCLErrors();
  }

 private:
  bool simulateError_;
};

class ProcessGroupCNCLSimulateErrors : public ProcessGroupCNCL {
 public:
  ProcessGroupCNCLSimulateErrors(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<ProcessGroupCNCL::Options> opts)
      : ProcessGroupCNCL(store, rank, size, opts), simulateError_(false) {}

  std::exception_ptr checkForCNCLErrors(
      std::shared_ptr<CNCLComm>& cnclComm) override {
    if (simulateError_) {
      return std::make_exception_ptr(std::runtime_error("Error"));
    }
    return ProcessGroupCNCL::checkForCNCLErrors(cnclComm);
  }

  std::chrono::duration<int64_t, std::milli> getWatchdogSleepInterval() {
    return std::chrono::milliseconds(
        ProcessGroupCNCLSimulateErrors::k_watchdog_thread_sleep_millis);
  }

  c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> initWork(
      at::Device device,
      int rank,
      c10d::OpType opType,
      const char* profilingTitle,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false) override {
    return c10::make_intrusive<WorkCNCLSimulateErrors>(
        device, simulateError_, rank, opType, seq_);
  }

  size_t getCNCLCommCacheSize() {
    return dev_cncl_comm_map_.size();
  }

  void simulateError() {
    simulateError_ = true;
  }

  void resetError() {
    simulateError_ = false;
  }

 private:
  bool simulateError_;
};

class WorkCNCLTimedoutErrors : public ProcessGroupCNCL::WorkCNCL {
 public:
  WorkCNCLTimedoutErrors(
      at::Device device,
      bool set_timedout_error,
      int rank,
      c10d::OpType opType,
      uint64_t seq)
      : WorkCNCL(device, rank, opType, seq),
        setTimedoutError_(set_timedout_error) {}

 private:
  bool isCompleted() override {
    if (setTimedoutError_) {
      return false;
    }
    return ProcessGroupCNCL::WorkCNCL::isCompleted();
  }

 private:
  bool setTimedoutError_;
};

class ProcessGroupCNCLTimedOutErrors : public ProcessGroupCNCLSimulateErrors {
 public:
  ProcessGroupCNCLTimedOutErrors(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<ProcessGroupCNCL::Options> opts)
      : ProcessGroupCNCLSimulateErrors(store, rank, size, opts),
        watchDogDebugInfoFinished_(false),
        setTimedoutError_(false) {}

  c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> initWork(
      at::Device device,
      int rank,
      c10d::OpType opType,
      const char* profilingTitle,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false) override {
    return c10::make_intrusive<WorkCNCLTimedoutErrors>(
        device, setTimedoutError_, rank, opType, seq_);
  }

  void setTimedoutError() {
    setTimedoutError_ = true;
  }

  void resetTimedoutError() {
    setTimedoutError_ = false;
  }

  bool getWatchDogDebugInfoFinishedFlag() {
    return watchDogDebugInfoFinished_;
  }

  // In the constructor of ProcessGroupCNCL. We don't allow the watchdog thread
  // to run any handling or desync report when the main thread is block wait.
  // Even if users set handling and turn on desyncDebug flag, they will get
  // reset. For the ease of unit test, we want the main thread to be block wait,
  // so we have this hack to manually set the desync debug flag after PG
  // creation.
  void forceSetDesyncDebugFlag() {
    desync_debug_ = true;
  }

 protected:
  std::string getCNCLWatchdogDebugInfo() override {
    LOG(INFO) << "overridden getCNCLWatchdogDebugInfo called";
    watchDogDebugInfoFinished_ = true;
    return "";
  }
  bool watchDogDebugInfoFinished_;

 private:
  bool setTimedoutError_;
};

class ProcessGroupCNCLNoHeartbeatCaught
    : public ProcessGroupCNCLTimedOutErrors {
 public:
  ProcessGroupCNCLNoHeartbeatCaught(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<ProcessGroupCNCL::Options> opts)
      : ProcessGroupCNCLTimedOutErrors(store, rank, size, opts),
        hasMonitorThreadCaughtError_(false) {}

  std::mutex& getWatchdogMutex() {
    return workMetaListMutex_;
  }

  bool getErrorCaughtFlag() {
    return hasMonitorThreadCaughtError_;
  }

  void forceTryWriteDebugInfo() {
    std::future<bool> asyncDebugDump = std::async(
        std::launch::async, [this]() { return this->dumpDebuggingInfo(); });
    asyncDebugDump.wait();
  }

 protected:
  // Override the heartbeat monitor function to make sure that we capture
  // the exception in the monitor thread because we cannot try-catch it in
  // the main thread and we set a flag for the main thread to check.
  void heartbeatMonitor() override {
    try {
      ProcessGroupCNCL::heartbeatMonitor();
    } catch (std::runtime_error& e) {
      hasMonitorThreadCaughtError_ = true;
    }
  }

  // It's really hard to unit test std::abort. So we override it instead.
  // Commented this override, we do see process aborted with core dump without
  // this override.
  void terminateProcess(std::string errMsg) override {
    throw std::runtime_error(errMsg);
  }

  bool hasMonitorThreadCaughtError_;
};

class ProcessGroupCNCLDebugInfoStuck
    : public ProcessGroupCNCLNoHeartbeatCaught {
 public:
  ProcessGroupCNCLDebugInfoStuck(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<ProcessGroupCNCL::Options> opts)
      : ProcessGroupCNCLNoHeartbeatCaught(store, rank, size, opts) {}

 protected:
  // Override the heartbeat monitor function to set a long timeout to mimic the
  // stuck in getting debug info.
  std::string getCNCLWatchdogDebugInfo() override {
    std::this_thread::sleep_for(
        std::chrono::seconds(heartbeatTimeoutInSec_ * 20));
    watchDogDebugInfoFinished_ = true;
    return "";
  }
};

class ProcessGroupCNCLErrorsTest : public ::testing::Test {
 protected:
  bool skipTest() {
    if (device_count() == 0) {
      LOG(INFO) << "Skipping test since MLU is not available";
      return true;
    }
    return false;
  }

  void SetUp() override {
    // Enable LOG(INFO) messages.
    c10::initLogging();
    // Need to have this check for at SetUp to make sure we only run the test --
    // including the init -- when there are MLUs available.
    if (skipTest()) {
      GTEST_SKIP() << "Skipping ProcessGroupCNCLErrorsTest because system "
                   << "requirement is not met (no MLU or MLU).";
    }

    size_t numDevices = 1; // One device per rank (thread)
    TemporaryFile file;
    store_ = c10::make_intrusive<c10d::FileStore>(file.path, 1);

    tensors_.resize(numDevices);
    tensors_[0] = at::empty({3, 3}, at::kPrivateUse1);
  }

  void TearDown() override {
    ASSERT_TRUE(setenv(TORCH_CNCL_BLOCKING_WAIT[0].c_str(), "0", 1) == 0);
  }

  std::vector<at::Tensor> tensors_;
  c10::intrusive_ptr<c10d::FileStore> store_;
};

TEST_F(ProcessGroupCNCLErrorsTest, testCNCLErrorsBlocking) {
  ASSERT_TRUE(setenv(TORCH_CNCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  auto options = ProcessGroupCNCL::Options::create();
  options->timeout = std::chrono::milliseconds(1000);
  ProcessGroupCNCLSimulateErrors pg(store_, 0, 1, options);

  auto work = pg.allreduce(tensors_);
  work->wait();
  EXPECT_EQ(1, pg.getCNCLCommCacheSize());

  // Now run all reduce with errors.
  pg.simulateError();
  work = pg.allreduce(tensors_);
  EXPECT_THROW(work->wait(), std::runtime_error);

  // Verify the work item failed.
  EXPECT_TRUE(work->isCompleted());
  EXPECT_THROW(work->wait(), std::runtime_error);

  // Communicators might be aborted here, further operations would fail.
}

TEST_F(ProcessGroupCNCLErrorsTest, testCNCLTimedoutErrorsBlocking) {
  ASSERT_TRUE(setenv(TORCH_CNCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  auto options = ProcessGroupCNCL::Options::create();
  options->timeout = std::chrono::milliseconds(3000);
  ProcessGroupCNCLTimedOutErrors pg(store_, 0, 1, options);

  auto work = pg.allreduce(tensors_);
  work->wait();
  EXPECT_EQ(1, pg.getCNCLCommCacheSize());

  // Now run all reduce with errors.
  pg.setTimedoutError();
  work = pg.allreduce(tensors_);
  EXPECT_THROW(work->wait(), c10::DistBackendError);

  // Communicators might be aborted here, further operations would fail.
}

TEST_F(ProcessGroupCNCLErrorsTest, testCNCLErrorsNonBlocking) {
  auto options = ProcessGroupCNCL::Options::create();
  options->timeout = std::chrono::milliseconds(3000);
  ProcessGroupCNCLSimulateErrors pg(store_, 0, 1, options);

  auto work = pg.allreduce(tensors_);
  pg.barrier()->wait();
  EXPECT_EQ(1, pg.getCNCLCommCacheSize());

  // Now run all reduce with errors.
  pg.simulateError();
  work = pg.allreduce(tensors_);

  // Should not throw exceptions.
  work->wait();
  pg.barrier()->wait();

  EXPECT_TRUE(work->isCompleted());
  // Communicators might be aborted here, further operations would fail.
}

// Function to read what we wrote to the local disk for validation.
std::string readTraceFromFile(const std::string& filename, size_t size) {
  std::ifstream file(filename, std::ios::binary);
  // Read the strings from the file
  if (file) { // While the file queue is in good state
    std::string str(size, '\0');
    file.read(&str[0], size);
    if (file) {
      return str;
    }
  }
  return "";
}

// Extend the nested class outside the parent class
class TestDebugInfoWriter : public DebugInfoWriter {
 public:
  TestDebugInfoWriter(std::string namePrefix)
      : DebugInfoWriter(namePrefix, 0) {}

  void write(const std::string& cnclTrace) override {
    traces_.assign(cnclTrace.begin(), cnclTrace.end());
    DebugInfoWriter::write(cnclTrace);
  }

  std::vector<uint8_t>& getTraces() {
    return traces_;
  }

 private:
  std::vector<uint8_t> traces_;
};

// TODO:[PYTORCH-11603]
 TEST_F(ProcessGroupCNCLErrorsTest, testCNCLErrorsNoHeartbeat) {
  int heartBeatIntervalInSec = 2;
  std::string timeInterval = std::to_string(heartBeatIntervalInSec);
  ASSERT_TRUE(setenv(TORCH_CNCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
  ASSERT_TRUE(
      setenv(
          TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC[0].c_str(),
          timeInterval.c_str(),
          1) == 0);
  ASSERT_TRUE(
      setenv(TORCH_CNCL_ENABLE_MONITORING[0].c_str(), "1", 1) == 0);
  auto tempFilename = c10::str(
      std::filesystem::temp_directory_path().string(), "/cncl_trace_rank_");
  ASSERT_TRUE(
      setenv("TORCH_CNCL_DEBUG_INFO_TEMP_FILE", tempFilename.c_str(), 1) ==
      0);
  // Enable cncl flight recorder.
  ASSERT_TRUE(setenv("TORCH_CNCL_TRACE_BUFFER_SIZE", "10", 1) == 0);
  auto options = ProcessGroupCNCL::Options::create();
  // Set a long watchdog timeout, so that we have enough time to lock the
  // watchdog and let the heartbeat monitor thread to kick in.
  options->timeout = std::chrono::milliseconds(30000);
  ProcessGroupCNCLNoHeartbeatCaught pg(store_, 0, 1, options);
  // The storer here is very similar to the fallback storer.
  // The only difference is that we are storing traces also in memory for
  // validation.
  std::string fileNamePrefix = getCvarString(
      {"TORCH_CNCL_DEBUG_INFO_TEMP_FILE"}, "/tmp/cncl_trace_rank_");
  std::unique_ptr<TestDebugInfoWriter> wrterForTestPtr =
      std::make_unique<TestDebugInfoWriter>(fileNamePrefix);
  std::vector<uint8_t>& traces = wrterForTestPtr->getTraces();
  DebugInfoWriter::registerWriter(std::move(wrterForTestPtr));

  // Normal collective case.
  auto work = pg.allreduce(tensors_);
  work->wait();

  work = pg.allreduce(tensors_);
  {
    // Now run all reduce with errors.
    std::lock_guard<std::mutex> lock(pg.getWatchdogMutex());
    LOG(INFO) << "Lock watchdog thread.";
    // Wait long enough before monitor thread throws exceptions.
    std::this_thread::sleep_for(
        std::chrono::seconds(heartBeatIntervalInSec * 3));
    // Check the monitoring thread launched and exception thrown.
    EXPECT_TRUE(pg.getErrorCaughtFlag());
  }
  work->wait();
  EXPECT_TRUE(traces.size() > 0);
  auto filename = c10::str(tempFilename, 0);
  auto traceFromStorage = readTraceFromFile(filename, traces.size());
  // Check the traces read from storage match with the original cncl trace.
  EXPECT_TRUE(traceFromStorage == std::string(traces.begin(), traces.end()));
  std::filesystem::remove(filename);
}

class ProcessGroupCNCLWatchdogTimeoutTest : public ProcessGroupCNCLErrorsTest {
 protected:
  void SetUp() override {
    // TODO (kwen2501)
    GTEST_SKIP() << "Skipping tests under ProcessGroupCNCLWatchdogTimeoutTest; "
                 << "will rewrite them after refactoring Work queues.";
    ProcessGroupCNCLErrorsTest::SetUp();
    std::string timeInterval = std::to_string(heartBeatIntervalInSec);
    ASSERT_TRUE(setenv(TORCH_CNCL_BLOCKING_WAIT[0].c_str(), "1", 1) == 0);
    ASSERT_TRUE(
        setenv(
            TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC[0].c_str(),
            timeInterval.c_str(),
            1) == 0);
    ASSERT_TRUE(setenv(TORCH_CNCL_ENABLE_MONITORING[0].c_str(), "1", 1) == 0);
    ASSERT_TRUE(setenv(TORCH_CNCL_DESYNC_DEBUG[0].c_str(), "1", 1) == 0);
    // We cannot capture the exception thrown in watchdog thread without making
    // lots of changes to the code. So we don't let the watchdog throw
    // exception.
    ASSERT_TRUE(
        setenv(TORCH_CNCL_ASYNC_ERROR_HANDLING[0].c_str(), "0", 1) == 0);
    options_ = ProcessGroupCNCL::Options::create();
    // Set a super short watchdog timeout.
    options_->timeout = std::chrono::milliseconds(100);
  }

  void watchdogTimeoutTestCommon(
      ProcessGroupCNCLNoHeartbeatCaught& pg,
      int multiplier) {
    pg.forceSetDesyncDebugFlag();
    pg.setTimedoutError();
    auto work = pg.allreduce(tensors_);
    std::this_thread::sleep_for(
        std::chrono::seconds(heartBeatIntervalInSec * multiplier));
    EXPECT_THROW(work->wait(), c10::DistBackendError);
  }

  const int heartBeatIntervalInSec = 2;
  c10::intrusive_ptr<ProcessGroupCNCL::Options> options_;
};

TEST_F(ProcessGroupCNCLWatchdogTimeoutTest, testCNCLTimedoutDebugInfoFinished) {
  ProcessGroupCNCLNoHeartbeatCaught pg(store_, 0, 1, options_);
  // Write debug info will lead to watchdog thread to wait for 30 seconds.
  // And this is hard to override, so we just call it before hand. Otherwise,
  // we need to set a long heartbeat timeout which will make the test way
  // slower.
  pg.forceTryWriteDebugInfo();
  watchdogTimeoutTestCommon(pg, 2);

  // The flag is true shows that the heartbeat monitor thread does not kill
  // the watchdog thread when it is getting debug info such as desync debug
  // info.
  EXPECT_TRUE(pg.getWatchDogDebugInfoFinishedFlag());
  // The flag is false shows that the heartbeat monitor thread does not
  // trigger process abort if getting debug info and destroy PG is fast.
  EXPECT_FALSE(pg.getErrorCaughtFlag());

  // Communicators might be aborted here, further operations would fail.
}

TEST_F(ProcessGroupCNCLWatchdogTimeoutTest, testCNCLTimedoutDebugInfoStuck) {
  ProcessGroupCNCLDebugInfoStuck pg(store_, 0, 1, options_);
  // Need to keep main thread sleep longer so that we can let heartbeat monitor
  // thread to finish the extra wait and flip the flag.
  watchdogTimeoutTestCommon(pg, 4);
  // The flag is false shows that we get stuck in getting debug info such as
  // desync debug info in the watchdog thread.
  EXPECT_FALSE(pg.getWatchDogDebugInfoFinishedFlag());
  // The flag is true shows that the heartbeat monitor thread does trigger
  // process abort if getting debug info gets stuck.
  EXPECT_TRUE(pg.getErrorCaughtFlag());

  // Communicators might be aborted here, further operations would fail.
}

} // namespace torch_mlu
