from __future__ import print_function

import os
import sys
import logging
import unittest
import numpy as np
import torch
import torch_mlu
import threading
import io
from itertools import product

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    TestCase,
    testinfo,
    freeze_rng_state,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestDevice(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_device(self):
        device_count = torch.mlu.device_count()
        for device_id in np.arange(0, device_count, 1):
            torch.mlu.set_device(device_id)
        input = torch.randn(8, 3, 24, 24).to(torch.mlu.current_device())
        input1 = torch.transpose(input, 0, 1)
        self.assertEqual(input1.device.index, torch.mlu.current_device())

    # @unittest.skip("not test")
    @testinfo()
    def test_device_by_tensor(self):
        torch.mlu.set_device(0)
        input = torch.randn(8, 3, 24, 24).to(torch.mlu.current_device())
        input1 = torch.transpose(input, 0, 1)
        input2 = torch.randn(4, 3, 24, 24).to(input1.device)
        self.assertEqual(input1.device.index, input2.device.index)

        device_count = torch.mlu.device_count()
        if device_count > 1:
            input_new = torch.randn(8, 3, 22, 22).to("mlu:0")
            input1_new = torch.transpose(input_new, 0, 1)
            input2_new = torch.randn(4, 3, 22, 22).to("mlu:1")
            self.assertNotEqual(input1_new.device.index, input2_new.device.index)

    # @unittest.skip("not test")
    @testinfo()
    def test_device_count(self):
        device_count = torch.mlu.device_count()
        self.assertLessEqual(0, device_count, "")

    # @unittest.skip("not test")
    @testinfo()
    def test_with_device(self):
        torch.mlu.set_device(0)
        self.assertEqual(0, torch.mlu.current_device())
        a = torch.randn(2, 2).mlu()
        for i in range(torch.mlu.device_count()):
            with torch.mlu.device(i):
                self.assertEqual(i, torch.mlu.current_device())
            with torch.mlu._DeviceGuard(i):
                self.assertEqual(i, torch.mlu.current_device())
            with torch.mlu.device_of(a):
                self.assertEqual(a.device.index, torch.mlu.current_device())
        self.assertEqual(0, torch.mlu.current_device())

    # @unittest.skip("not test")
    @testinfo()
    def test_device_synchronize(self):
        torch.mlu.synchronize()
        torch.mlu.synchronize("mlu")
        torch.mlu.synchronize("mlu:0")
        for i in range(torch.mlu.device_count()):
            torch.mlu.synchronize(i)

        with self.assertRaisesRegex(ValueError, "Expected a mlu device, but got: cpu"):
            torch.mlu.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected a mlu device, but got: cpu"):
            torch.mlu.synchronize("cpu")

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_device(self):
        if torch.mlu.device_count() <= 1:
            return
        x = torch.randn(5, 5).mlu()
        with torch.mlu.device(1):
            y = x.mlu()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.mlu(), y)
            z = y.mlu(0)
            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.mlu(0), z)

        x = torch.randn(5, 5)
        with torch.mlu.device(1):
            y = x.mlu()
            self.assertEqual(y.get_device(), 1)
            self.assertIs(y.mlu(), y)
            z = y.mlu(0)
            self.assertEqual(z.get_device(), 0)
            self.assertIs(z.mlu(0), z)

    # @unittest.skip("not test")
    @testinfo()
    def test_cublas_multiple_threads_same_device(self):
        # Note, these parameters should be very carefully tuned
        # Too small number makes it hard for the racing condition
        # to happen, while too large number sometimes cause hang
        size = 1024
        num_threads = 2
        trials = 3
        test_iters = 100

        weight = torch.ones((size, size), device="mlu")
        results = {}
        barrier = threading.Barrier(num_threads)

        def _worker(t):
            my_stream = torch.mlu.Stream()
            # Hard sync so we don't need to worry about creating and using tensors
            # across streams or the fact that default streams are thread-local.
            # Those issues are not the target of this test.
            torch.mlu.synchronize()
            # Line up threads to increase likelihood of race conditions.
            barrier.wait()
            with torch.mlu.stream(my_stream):
                for i in range(test_iters):
                    # If all threads are sharing the same cublas handle,
                    # the following sequence may occur:
                    # thread 0 calls cublasSetStream()
                    # thread 1 calls cublasSetStream()
                    # thread 0 launches its raw gemm, which it thinks is in
                    #          its own stream, but is actually in thread 1's stream.
                    # thread 0 enqueues its div_, which IS is its own stream,
                    #          but actually now races with its gemm.
                    results[t] = torch.mm(results[t], weight)
                    results[t].div_(float(size))
            torch.mlu.synchronize()

        for _ in range(trials):
            for t in range(num_threads):
                results[t] = torch.ones((size, size), device="mlu")

            threads = [
                threading.Thread(target=_worker, args=(t,)) for t in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for t in range(num_threads):
                self.assertEqual(results[t].sum().item(), size * size)

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_device_memory_allocated(self):
        from torch.mlu import memory_allocated

        device_count = torch.mlu.device_count()
        current_alloc = [memory_allocated(idx) for idx in range(device_count)]
        x = torch.ones(10, device="mlu:0")
        self.assertTrue(memory_allocated(0) > current_alloc[0])
        self.assertTrue(
            all(
                memory_allocated(torch.mlu.device(idx)) == current_alloc[idx]
                for idx in range(1, device_count)
            )
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_get_device_capability(self):
        # Testing the behaviour with None as an argument
        current_device = torch.mlu.current_device()
        current_device_capability = torch.mlu.get_device_capability(current_device)
        device_capability_None = torch.mlu.get_device_capability(None)
        self.assertEqual(current_device_capability, device_capability_None)

        # Testing the behaviour for No argument
        device_capability_no_argument = torch.mlu.get_device_capability()
        self.assertEqual(current_device_capability, device_capability_no_argument)

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_get_device_name(self):
        # Testing the behaviour with None as an argument
        current_device = torch.mlu.current_device()
        current_device_name = torch.mlu.get_device_name(current_device)
        device_name_None = torch.mlu.get_device_name(None)
        self.assertEqual(current_device_name, device_name_None)

        # Testing the behaviour for No argument
        device_name_no_argument = torch.mlu.get_device_name()
        self.assertEqual(current_device_name, device_name_no_argument)

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_set_device(self):
        if torch.mlu.device_count() <= 1:
            return
        x = torch.randn(5, 5)
        with torch.mlu.device(1):
            self.assertEqual(x.mlu().get_device(), 1)
            torch.mlu.set_device(0)
            self.assertEqual(x.mlu().get_device(), 0)
            with torch.mlu.device(1):
                self.assertEqual(x.mlu().get_device(), 1)
            self.assertEqual(x.mlu().get_device(), 0)
            torch.mlu.set_device(1)
        self.assertEqual(x.mlu().get_device(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_cudnn_multiple_threads_same_device(self):
        # This function is intended to test the lazy creation and reuse of per-thread
        # cudnn handles on each device in aten/src/ATen/cudnn/Handles.cpp.
        # Failure here likely indicates something wrong with that logic.
        weight = torch.ones((1, 1, 2, 2), device="mlu")

        results = {}

        num_threads = 2
        trials = 3
        test_iters = 1000
        barrier = threading.Barrier(num_threads)

        with torch.backends.cudnn.flags(enabled=True):

            def _worker(t):
                my_stream = torch.mlu.Stream()
                # Hard sync so we don't need to worry about creating and using tensors
                # across streams or the fact that default streams are thread-local.
                # Those issues are not the target of this test.
                torch.mlu.synchronize()
                # Line up threads to increase likelihood of race conditions.
                barrier.wait()
                with torch.mlu.stream(my_stream):
                    for _ in range(test_iters):
                        # If all threads are sharing the same cudnn handle,
                        # the following sequence may occur:
                        # thread 0 calls setCuDNNStreamToCurrent()
                        # thread 1 calls setCuDNNStreamToCurrent()
                        # thread 0 launches its raw convolution, which it thinks is in
                        #          its own stream, but is actually in thread 1's stream.
                        # thread 0 enqueues its div_, which IS is its own stream,
                        #          but now races with its convolution.
                        results[t] = torch.nn.functional.conv2d(
                            results[t], weight, padding=0
                        )
                        results[t].div_(4.0)
                torch.mlu.synchronize()

            for _ in range(trials):
                for t in range(num_threads):
                    results[t] = torch.ones((1, 1, 2048, 2048), device="mlu")

                threads = [
                    threading.Thread(target=_worker, args=(t,))
                    for t in range(num_threads)
                ]

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                for t in range(num_threads):
                    self.assertEqual(
                        results[t].sum().item(),
                        (2048 - test_iters) * (2048 - test_iters),
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_get_device_index(self):
        from torch_mlu.mlu._utils import _get_device_index

        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            _get_device_index("mlu0", optional=True)

        with self.assertRaisesRegex(ValueError, "Expected a mlu device"):
            cpu_device = torch.device("cpu")
            _get_device_index(cpu_device, optional=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_grad_scaling_device_as_key(self):
        if torch.mlu.device_count() <= 1:
            return
        # Ensure that different instances of "device" objects that point to the same device
        # are treated as identical keys by dicts.  GradScaler relies on this behavior, and may
        # error otherwise in a way that's difficult to detect (a silent performance hit).
        d = {}
        t = torch.empty((1,), device="mlu:0")
        dev0a = torch.device("mlu:0")
        dev0b = torch.device("mlu:0")
        dev1a = torch.device("mlu:1")
        dev1b = torch.device("mlu:1")

        self.assertTrue(hash(dev0a) == hash(dev0b))
        self.assertTrue(hash(dev1a) == hash(dev1b))

        d[dev0a] = "0a"
        d[dev0b] = "0b"
        self.assertTrue(len(d) == 1)
        self.assertTrue(d[dev0a] == "0b")
        d[t.device] = "t"
        self.assertTrue(len(d) == 1)
        self.assertTrue(d[dev0a] == "t")

        d[dev1a] = "1a"
        d[dev1b] = "1b"
        self.assertTrue(len(d) == 2)
        self.assertTrue(d[dev1a] == "1b")

    # @unittest.skip("not test")
    @testinfo()
    def test_load_nonexistent_device(self):
        if torch.mlu.device_count() >= 10:
            return
        # Setup: create a serialized file object with a 'mlu:9' restore location
        tensor = torch.randn(2, device="mlu")
        buf = io.BytesIO()
        torch.save(tensor, buf)
        # NB: this might not work in the future if serialization changes
        buf = io.BytesIO(buf.getvalue().replace(b"mlu:0", b"mlu:9"))

        msg = r"Attempting to deserialize object on MLU device 9"
        with self.assertRaisesRegex(RuntimeError, msg):
            _ = torch.load(buf)

    # @unittest.skip("not test")
    @testinfo()
    def test_specify_improper_device_name(self):
        import os

        fname = "tempfile.pt"
        try:
            with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
                torch.save(
                    [torch.nn.Parameter(torch.randn(10, 10))],
                    fname,
                    _use_new_zipfile_serialization=True,
                )
                torch.load(fname, "mlu0")
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    # @unittest.skip("not test")
    @testinfo()
    def test_stream_event_device(self):
        if torch.mlu.device_count() <= 1:
            return
        d0 = torch.device("mlu:0")
        d1 = torch.device("mlu:1")
        e0 = torch.mlu.Event()

        self.assertEqual(None, e0.device)

        with torch.mlu.device(d0):
            s0 = torch.mlu.current_stream()
            s0.record_event(e0)

        with torch.mlu.device(d1):
            s1 = torch.mlu.Stream()
            e1 = s1.record_event()

        self.assertEqual(s0.device, torch.device("mlu:0"))
        self.assertEqual(e0.device, torch.device("mlu:0"))
        self.assertEqual(s1.device, torch.device("mlu:1"))
        self.assertEqual(e1.device, torch.device("mlu:1"))

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_device(self):
        if torch.mlu.device_count() <= 1:
            return
        self.assertEqual(torch.mlu.FloatTensor(1).get_device(), 0)
        self.assertEqual(torch.mlu.FloatTensor(1, device=1).get_device(), 1)
        with torch.mlu.device(1):
            self.assertEqual(torch.mlu.FloatTensor(1).get_device(), 1)
            self.assertEqual(torch.mlu.FloatTensor(1, device=0).get_device(), 0)
            self.assertEqual(torch.mlu.FloatTensor(1, device=None).get_device(), 1)

    # @unittest.skip("not test")
    @testinfo()
    def test_torch_manual_seed_seeds_mlu_devices(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().mlu()
            torch.manual_seed(2)
            self.assertEqual(torch.mlu.initial_seed(), 2)
            x.uniform_()
            torch.manual_seed(2)
            y = x.clone().uniform_()
            self.assertEqual(x, y)
            self.assertEqual(torch.mlu.initial_seed(), 2)

    # @unittest.skip("not test")
    @testinfo()
    def test_matmul_device_mismatch(self):
        cpu = torch.rand((10, 10))
        mlu = cpu.mlu()
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            cpu @ mlu
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            mlu @ cpu

        for s, m1, m2 in product((cpu, mlu), repeat=3):
            if s.device == m1.device == m2.device:
                torch.addmm(s, m1, m2)
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "Expected all tensors to be on the same device"
                ):
                    torch.addmm(s, m1, m2)

    # @unittest.skip("not test")
    @testinfo()
    def test_check_device_access_peer_ability(self):
        device_counts = torch.mlu.device_count()
        if device_counts < 2:
            return
        from random import sample

        device_indexs = [idx for idx in range(device_counts)]
        index_samples = sample(device_indexs, 2)
        # Just call the can_device_access_peer API, to ensure that there are no errors occur.
        torch.mlu.can_device_access_peer(index_samples[0], index_samples[1])


if __name__ == "__main__":
    unittest.main()
