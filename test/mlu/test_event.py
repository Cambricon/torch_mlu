# pylint: disable=W0612
from __future__ import print_function

import os
import subprocess
import sys
import unittest
import logging
import time
import torch
import torch_mlu
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import NO_MULTIPROCESSING_SPAWN

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    run_tests,
    get_cycles_per_ms,
)  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)

TEST_MLU_EVENT_IPC = (
    torch.mlu.is_available() and sys.platform != "darwin" and sys.platform != "win32"
)
TEST_MULTIMLU = TEST_MLU_EVENT_IPC and torch.mlu.device_count() > 1


# Multiply by two in a separate stream
def mlu_multiply_two(queue, ready, done):
    ready.set()
    with torch.mlu.stream(torch.mlu.Stream()):
        mlu_event, tensor = queue.get()
        mlu_event.wait()
        tensor.mul_(2)
        mlu_event.record()
        done.set()
        del mlu_event


class TestEvent(TestCase):
    @testinfo()
    def test_record(self):  # pylint: disable=R0022
        input1 = torch.randn(1, 3, 2, 2).to("mlu")
        output = torch.neg(input1)
        event = torch.mlu.Event()
        event.record()

    @testinfo()
    def test_query(self):
        event = torch.mlu.Event()
        self.assertTrue(event.query())

    @testinfo()
    def test_synchronize_enable_timing(self):
        input1 = torch.randn(1000, 1000, 2, 2).to("mlu")
        input2 = torch.randn(1000, 1000, 2, 2).to("mlu")
        output = torch.neg(input1)
        start = torch.mlu.Event(enable_timing=True)
        end = torch.mlu.Event(enable_timing=True)
        start.record()
        for i in range(10):
            input3 = torch.neg(input1)
            input4 = torch.neg(input2)
        end.record()
        end.synchronize()
        e2e_time_ms = start.elapsed_time(end)
        hardware_time_ms = start.hardware_time(end) / 1000.0
        diff_ms = e2e_time_ms - hardware_time_ms
        self.assertTrue(diff_ms >= 0)

    @testinfo()
    def test_synchronize_blocking(self):
        def _test_event_sync(blocking_flag):
            start = torch.mlu.Event(blocking=blocking_flag)
            end = torch.mlu.Event(blocking=blocking_flag)

            start.record()
            torch.mlu._sleep(int(5000 * get_cycles_per_ms()))
            end.record()

            start_wall = time.time()
            start_cpu = time.process_time()
            end.synchronize()
            end_wall = time.time()
            end_cpu = time.process_time()

            wall_time = end_wall - start_wall
            cpu_time = end_cpu - start_cpu
            print(f"wall_time = {wall_time:.3f}s, cpu_time = {cpu_time:.3f}s")

            return wall_time, cpu_time

        wall_block, cpu_block = _test_event_sync(True)
        ratio_block = (cpu_block / wall_block * 100) if wall_block > 0 else 0
        _, _ = _test_event_sync(False)
        # In theory, if the blocking mode fully puts the thread to sleep,
        # the CPU time consumed during synchronization should be nearly zero,
        # resulting in a ratio of (CPU time / wall time) close to 0%.
        # However, due to OS scheduling, measurement overhead, and other factors,
        # we relax the threshold: in our test, we expect the CPU time in blocking mode
        # to be significantly lower than that in non-blocking mode.
        self.assertLess(
            ratio_block,
            10,
            f"Blocking mode expected CPU time ratio < 10%, but got {ratio_block:.2f}%",
        )

    @testinfo()
    def test_wait(self):
        start = torch.mlu.Event()
        queue = torch.mlu.current_stream()
        user_queue = torch.mlu.Stream()
        torch.mlu._sleep(int(800 * get_cycles_per_ms()))
        start.record(queue)
        start.wait(user_queue)
        with torch.mlu.stream(user_queue):
            torch.mlu._sleep(int(20 * get_cycles_per_ms()))
        user_queue.synchronize()
        self.assertTrue(start.query())
        self.assertTrue(queue.query())
        self.assertTrue(user_queue.query())

    @testinfo()
    def test_event_repr(self):
        e = torch.mlu.Event()
        self.assertTrue("torch.mlu.Event" in e.__repr__())

    @testinfo()
    def test_generic_event_without_create_context(self):
        if torch.mlu.device_count() < 3:
            return
        torch.mlu.set_device(0)
        a = torch.randn(3, 3).mlu()
        start = torch.mlu.Event(enable_timing=True)
        end = torch.mlu.Event(enable_timing=True)
        start.record()
        a.add(a)
        end.record()
        torch.mlu.set_device(2)
        self.assertTrue(torch_mlu._MLUC._mlu_hasPrimaryContext(0))
        self.assertFalse(torch_mlu._MLUC._mlu_hasPrimaryContext(2))
        elp_time = start.elapsed_time(end)
        self.assertTrue(torch_mlu._MLUC._mlu_hasPrimaryContext(0))
        self.assertFalse(torch_mlu._MLUC._mlu_hasPrimaryContext(2))


if __name__ == "__main__":
    unittest.main()
