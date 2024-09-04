#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <unordered_set>
#include <c10/util/Logging.h>
#include "framework/core/device.h"
#include "framework/core/MLUEvent.h"
#include "framework/core/MLUStream.h"
#include "c10/util/Optional.h"
#include "cnrt.h"

namespace torch_mlu {

struct ipc_sample {
  cnrtIpcMemHandle m_handle;
  cnrtIpcNotifierHandle n_handle;
  int start;
};

TEST(MLUEventTest, ipc_handle) {
  unsigned int flags =
      CNRT_NOTIFIER_INTERPROCESS | CNRT_NOTIFIER_DISABLE_TIMING_ALL;
  MLUEvent mlu_event(flags);
  void* dev_mem;
  char* host_mem;
  size_t mem_size = 4 * sizeof(char);

  // shared info between child and parent.
  struct ipc_sample* info = (struct ipc_sample*)mmap(
      NULL,
      sizeof(struct ipc_sample),
      PROT_READ | PROT_WRITE,
      MAP_SHARED | MAP_ANONYMOUS,
      0,
      0);
  info->start = 0;
  pid_t pid = fork();

  int major = -1;
  int device_id;
  cnrtGetDevice(&device_id);
  TORCH_CNRT_CHECK(cnrtDeviceGetAttribute(
      &major, cnrtAttrComputeCapabilityMajor, device_id));
  if (major != 5)
    GTEST_SKIP()
        << "Skipping: IPC Handle does not supported on current device.";

  if (pid < 0) {
    printf("error fork\n");
  } else if (pid == 0) {
    printf("child process..\n");
    host_mem = (char*)calloc(1, mem_size);
    memset(host_mem, 'B', mem_size);
    while (0 == info->start) {
      sched_yield();
    } // wait until parent process set IPC handle.
    TORCH_CNRT_CHECK(cnrtMapMemHandle(&dev_mem, info->m_handle, 0));
    MLUEvent mlu_event_new(mlu_event.device_index(), &info->n_handle);
    mlu_event_new.wait(getCurrentMLUStream());
    TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
        (void*)host_mem,
        dev_mem,
        mem_size,
        getCurrentMLUStream(),
        cnrtMemcpyDevToHost));
    sleep(5); // wait for cp complete.
    ASSERT_TRUE(*host_mem == 'A'); // parent process set host_mem to 'A'.
    TORCH_CNRT_CHECK(cnrtUnMapMemHandle(dev_mem));
    free(host_mem);
    exit(testing::Test::HasFailure());
  } else {
    printf("parent process.. child's pid is %d\n", pid);
    host_mem = (char*)calloc(1, mem_size);
    memset(host_mem, 'A', mem_size);
    mlu_event.ipc_handle(&info->n_handle);
    TORCH_CNRT_CHECK(cnrtMalloc(&dev_mem, mem_size));
    TORCH_CNRT_CHECK(cnrtAcquireMemHandle(&info->m_handle, dev_mem));
    TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
        dev_mem,
        (void*)host_mem,
        mem_size,
        getCurrentMLUStream(),
        cnrtMemcpyHostToDev));
    mlu_event.place();
    __sync_add_and_fetch(
        &info->start, 1); // tell child process IPC handle is ready.
    int status = -1;
    if (waitpid(pid, &status, 0) < 0) {
      printf("%s, waitpid error.\n", __func__);
      exit(EXIT_FAILURE);
    }
    EXPECT_EQ(WEXITSTATUS(status), EXIT_SUCCESS);
    EXPECT_NE(WIFEXITED(status), 0);
    TORCH_CNRT_CHECK(cnrtFree(dev_mem));
    free(host_mem);
  }
}

TEST(MLUEventTest, placeMLUEvent) {
  MLUEvent mlu_event;
  mlu_event.place();
  auto stream = getStreamFromPool();
  mlu_event.place(stream);
}

TEST(MLUEventTest, syncMLUEvent) {
  MLUEvent mlu_event;
  mlu_event.place();
  mlu_event.synchronize();
}

TEST(MLUEventTest, elapsed_time) {
  MLUEvent start(CNRT_NOTIFIER_DEFAULT);
  MLUEvent end(CNRT_NOTIFIER_DEFAULT);
  start.place();
  end.place();
  end.synchronize();
  float time = start.elapsed_time(end);
  ASSERT_TRUE(time >= 0);
}

TEST(MLUEventTest, hardware_time) {
  MLUEvent start(CNRT_NOTIFIER_DEFAULT);
  MLUEvent end(CNRT_NOTIFIER_DEFAULT);
  start.place();
  end.place();
  end.synchronize();
  float time = start.hardware_time(end);
  ASSERT_TRUE(time >= 0);
}

TEST(MLUEventTest, stream_wait_mlu_event) {
  MLUEvent mlu_event;
  mlu_event.place();
  mlu_event.wait(getCurrentMLUStream());
  mlu_event.synchronize();
  mlu_event.place();
  mlu_event.wait(getStreamFromPool());
  mlu_event.synchronize();
}

TEST(MLUEventTest, query_and_wait_mlu_event) {
  MLUEvent mlu_event;
  ASSERT_TRUE(mlu_event.query());
  mlu_event.synchronize();
  mlu_event.place();
  mlu_event.synchronize();
  ASSERT_TRUE(mlu_event.query());
}

TEST(MLUEventTest, move_test) {
  MLUEvent no_1;
  MLUEvent no_2;
  no_2 = std::move(no_1);
  TORCH_CHECK_EQ(no_1.device_index(), no_2.device_index());
  TORCH_CHECK_EQ(no_1.isCreated(), no_2.isCreated());

  MLUEvent start = std::move(MLUEvent(CNRT_NOTIFIER_DEFAULT));
  MLUEvent end = std::move(MLUEvent(CNRT_NOTIFIER_DEFAULT));
  start.place();
  end.place();
  end.synchronize();
  float time = start.elapsed_time(end);
  ASSERT_TRUE(time >= 0);
}

} // namespace torch_mlu
