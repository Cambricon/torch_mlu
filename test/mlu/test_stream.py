from __future__ import print_function
import os
import sys
import unittest
import threading
import logging
import contextlib
import torch
import torch_mlu  # pylint: disable=W0611
from torch.nn import Parameter  # pylint: disable=W0611, C0411
import torch.nn.functional as F  # pylint: disable=W0611, C0411

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411, C0413

logging.basicConfig(level=logging.DEBUG)


class TestStream(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_stream(self):
        default_stream = torch.mlu.current_stream()
        user_stream = torch.mlu.Stream()
        self.assertEqual(torch.mlu.current_stream(), default_stream)
        self.assertNotEqual(default_stream, user_stream)
        # self.assertEqual(default_stream.mlu_stream, 0)
        self.assertNotEqual(user_stream.mlu_stream, 0)
        with torch.mlu.stream(user_stream):
            self.assertEqual(torch.mlu.current_stream(), user_stream)
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        tensor2 = tensor1.mlu(non_blocking=True) + 1  # pylint: disable=W0612
        default_stream.synchronize()
        self.assertTrue(default_stream.query())

    # @unittest.skip("not test")
    @testinfo()
    def test_to_no_blocking(self):
        def _test_to_non_blocking(a, non_blocking):
            stream = torch.mlu.current_stream()
            with torch.mlu.stream(stream):
                b = a.to("mlu", non_blocking=non_blocking)
                self.assertEqual(stream.query(), not non_blocking)
                stream.synchronize()
                self.assertEqual(a, b)

        # 10MB copies
        x = torch.ones(100000000, dtype=torch.uint8).pin_memory()
        _test_to_non_blocking(x, True)
        y = torch.ones(10000000, dtype=torch.uint8)
        _test_to_non_blocking(y, False)

    # @unittest.skip("not test")
    @testinfo()
    def test_current_stream(self):
        if torch.mlu.device_count() <= 1:
            return
        d0 = torch.device("mlu:0")
        d1 = torch.device("mlu:1")
        s0 = torch.mlu.current_stream()
        s1 = torch.mlu.current_stream(device=1)
        s2 = torch.mlu.current_stream(device=0)
        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s2)
        with torch.mlu.device(d1):
            s0 = torch.mlu.current_stream()
            s1 = torch.mlu.current_stream(1)
            s2 = torch.mlu.current_stream(d0)
        self.assertEqual(d1, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(s0, s1)

    # @unittest.skip("not test")
    @testinfo()
    def test_streams_multi_mlu(self):
        if torch.mlu.device_count() <= 1:
            return
        default_stream = torch.mlu.current_stream()
        self.assertEqual(default_stream.device, torch.device("mlu:0"))
        stream = torch.mlu.Stream(device=1)
        self.assertEqual(stream.device, torch.device("mlu:1"))
        with torch.mlu.device("mlu:1"):
            self.assertEqual(torch.mlu.current_stream().device, torch.device("mlu:1"))
            self.assertNotEqual(torch.mlu.current_stream(), default_stream)

    # @unittest.skip("not test")
    @testinfo()
    def test_stream_context(self):
        if torch.mlu.device_count() <= 1:
            return
        s0 = torch.mlu.current_stream()
        s1 = torch.mlu.Stream(device=1)
        s2 = torch.mlu.Stream(device=0)
        with torch.mlu.device(s1.device):
            prev_stream_on_mlu1 = torch.mlu.current_stream()
        self.assertEqual(torch.mlu.current_stream(), s0)
        self.assertEqual(0, torch.mlu.current_device())
        with torch.mlu.stream(s1):
            self.assertEqual(torch.mlu.current_stream().mlu_stream, s1.mlu_stream)
            self.assertEqual(1, torch.mlu.current_device())
            with torch.mlu.stream(s2):
                self.assertEqual(torch.mlu.current_stream().mlu_stream, s2.mlu_stream)
                self.assertEqual(0, torch.mlu.current_device())
                with torch.mlu.stream(s0):
                    self.assertEqual(
                        torch.mlu.current_stream().mlu_stream, s0.mlu_stream
                    )
                    self.assertEqual(0, torch.mlu.current_device())
                self.assertEqual(torch.mlu.current_stream().mlu_stream, s2.mlu_stream)
                self.assertEqual(0, torch.mlu.current_device())
            self.assertEqual(torch.mlu.current_stream().mlu_stream, s1.mlu_stream)
            self.assertEqual(1, torch.mlu.current_device())
        with torch.mlu.device(s1.device):
            self.assertEqual(prev_stream_on_mlu1, torch.mlu.current_stream())
        self.assertEqual(torch.mlu.current_stream(), s0)
        self.assertEqual(0, torch.mlu.current_device())

    # @unittest.skip("not test")
    @testinfo()
    def test_streams_multi_mlu_eq(self):
        if torch.mlu.device_count() <= 1:
            return
        d0 = torch.device("mlu:0")
        d1 = torch.device("mlu:1")
        with torch.mlu.device(d0):
            s0 = torch.mlu.current_stream()
            s1 = torch.mlu.current_stream()
        with torch.mlu.device(d1):
            s2 = torch.mlu.current_stream()
            s3 = torch.mlu.current_stream()
        self.assertTrue(s0 == s0)  # pylint: disable=R0124
        self.assertTrue(s0 == s1)
        self.assertTrue(s2 == s2)  # pylint: disable=R0124
        self.assertTrue(s2 == s3)
        self.assertFalse(s0 == s2)
        self.assertFalse(s1 == s3)
        self.assertEqual(s0.device, s1.device)
        self.assertEqual(s0.mlu_stream, s1.mlu_stream)
        self.assertEqual(s2.device, s3.device)
        self.assertEqual(s2.mlu_stream, s3.mlu_stream)
        self.assertNotEqual(s0.device, s3.device)

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
            e1 = s1.record_event()  # pylint: disable=W0612
        self.assertEqual(s0.device, torch.device("mlu:0"))
        self.assertEqual(e0.device, torch.device("mlu:0"))
        self.assertEqual(s1.device, torch.device("mlu:1"))
        self.assertEqual(e1.device, torch.device("mlu:1"))

    # @unittest.skip("not test")
    @testinfo()
    def test_streaming_backwards_sync(self):
        default_stream = torch.mlu.current_stream()
        stream = torch.mlu.Stream()
        input1 = torch.zeros((1024, 10240), dtype=torch.float32, device="mlu")
        input2 = torch.zeros((10240, 1024), dtype=torch.float32, device="mlu")

        class MultiplyInStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(torch.mlu.current_stream(), stream)
                # delays the operation in the the background stream
                # torch.mlu._sleep(1000 * 1000)
                # torch.matmul instead of torch.mlu._sleep
                torch.matmul(input1, input2)
                return grad * 2

        x = torch.randn(5, 5, device="mlu", requires_grad=True)
        with torch.mlu.stream(stream):
            stream.wait_stream(default_stream)
            output = MultiplyInStream.apply(x)
            output.sum().backward()
        # sync needed
        default_stream.wait_stream(stream)
        self.assertEqual(x.grad, torch.ones_like(x) * 2)
        self.assertEqual(torch.mlu.current_stream(), default_stream)

    # @unittest.skip("not test")
    @testinfo()
    def test_default_stream(self):
        if torch.mlu.device_count() <= 1:
            return
        d0 = torch.device("mlu:0")
        d1 = torch.device("mlu:1")

        with torch.mlu.device(d0):
            s0 = torch.mlu.default_stream()

        with torch.mlu.device(d1):
            s1 = torch.mlu.default_stream()

        s2 = torch.mlu.default_stream(device=0)
        s3 = torch.mlu.default_stream(d1)

        self.assertEqual(d0, s0.device)
        self.assertEqual(d1, s1.device)
        self.assertEqual(d0, s2.device)
        self.assertEqual(d1, s3.device)
        self.assertEqual(s0, s2)
        self.assertEqual(s1, s3)

        with torch.mlu.device(d0):
            self.assertEqual(torch.mlu.current_stream(), s0)

        with torch.mlu.device(d1):
            self.assertEqual(torch.mlu.current_stream(), s1)

        new_stream = torch.mlu.Stream()
        assert torch.mlu.current_stream() == torch.mlu.default_stream()
        with torch.mlu.stream(new_stream):
            assert torch.mlu.current_stream() != torch.mlu.default_stream()

    def _test_copy_sync_current_stream(self, x, y):
        x_plus_one = x + 1
        s0 = torch.mlu.Stream(device=x.device)
        s1 = torch.mlu.Stream(device=y.device)
        s2 = torch.mlu.Stream(device=x.device)
        s3 = torch.mlu.Stream(device=y.device)

        # same dst stream different src streams
        with torch.mlu.stream(s0):
            # torch.mlu._sleep(TestCuda.FIFTY_MIL_CYCLES)
            with torch.mlu.stream(s1):
                y.copy_(x_plus_one)

        with torch.mlu.stream(s2), torch.mlu.stream(s1):
            y.copy_(x)

        s1.synchronize()
        # The copy() is synchronized on the current streams of both src and dst.
        # In the above test, the _sleep() op on s0 will not block the copy() on
        # s2, but both copies are synchronized on s1 in the dst device. Hence,
        # x is copied to y after x_plus_one is copied to y. If x and y are on
        # the same device, both copy() ops are synchronized on s1.
        self.assertEqual(y, x)

        # same src stream different dst streams
        with torch.mlu.stream(s1):
            # torch.mlu._sleep(TestCuda.FIFTY_MIL_CYCLES)
            with torch.mlu.stream(s0):
                y.copy_(x_plus_one)

        with torch.mlu.stream(s3), torch.mlu.stream(s0):
            y.copy_(x)

        s0.synchronize()
        # Similarly, both copy() ops are synchronized on s0.
        self.assertEqual(y, x)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_streams(self):
        if torch.mlu.device_count() <= 1:
            return
        d0 = torch.device("mlu:0")
        x0 = torch.zeros(5, 5, device=d0)
        d1 = torch.device("mlu:1")
        x1 = torch.zeros(5, 5, device=d1)
        self._test_copy_sync_current_stream(x0, x1)
        x2 = torch.zeros(5, 5, device=d0)
        self._test_copy_sync_current_stream(x0, x2)

    # @unittest.skip("not test")
    @testinfo()
    def test_stream_repeatedly(self):
        if torch.mlu.device_count() <= 1:
            return
        for i in range(100):
            user_stream = torch.mlu.Stream(device=1)
            with torch.mlu.stream(user_stream):
                x = torch.randn((32, 3, 24, 24), dtype=torch.float32)
                x_mlu = x.to(torch.device("mlu"))
                out = torch.abs(x_mlu)
            out1 = torch.cat((out, out))

    # @unittest.skip("not test")
    @testinfo()
    def test_streams_priority(self):
        low, high = torch.mlu.Stream.priority_range()
        for p in range(high, low):
            s0 = torch.mlu.Stream(device=0, priority=p)
            self.assertEqual(p, s0.priority)
            self.assertEqual(torch.device("mlu:0"), s0.device)
            if torch.mlu.device_count() <= 1:
                continue
            s1 = torch.mlu.Stream(device=1, priority=p)
            self.assertEqual(p, s1.priority)
            self.assertEqual(torch.device("mlu:1"), s1.device)

    @unittest.skip("need align for native PT1.13.1 update or delete directly")
    @testinfo()
    def test_streaming_backwards_multiple_streams(self):
        class StreamModel(torch.nn.Module):
            def __init__(self):
                super(StreamModel, self).__init__()
                self.event = torch.mlu.Event()
                self.stream0 = torch.mlu.Stream()
                self.stream1 = torch.mlu.Stream()

            def forward(self, x):
                x0 = x.clone()
                torch_mlu._MLUC._mlu_setStream(self.stream0._cdata)
                y0 = x0 * 2
                self.event.record(stream=torch.mlu.current_stream())

                torch_mlu._MLUC._mlu_setStream(self.stream1._cdata)
                y1 = x * 3
                self.stream1.wait_event(self.event)
                return y0 + y1

        stream = torch.mlu.Stream()

        def accum_hook(grad):
            self.assertEqual(torch.mlu.current_stream(), stream)

        with torch.mlu.stream(stream):
            x = torch.ones(5, 5, device="mlu", requires_grad=True)
            x.register_hook(accum_hook)
            torch.mlu.current_stream().wait_stream(stream)
            model = StreamModel().mlu()
            model(x).sum().backward()

        self.assertEqual(x.grad, torch.ones_like(x) * 5)

    # @unittest.skip("not test")
    @testinfo()
    def test_multiple_threads_same_device(self):
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
    def test_streaming_backwards_device_transfer(self):
        if torch.mlu.device_count() <= 1:
            return
        # This function must run with non-default current streams on all devices, otherwise it's meaningless.
        # The intention is to test that to()'s backward (CopyBackward) interacts properly with the
        # synchronization logic in torch/csrc/autograd/input_buffer.cpp.
        dev0 = torch.device("mlu:0")
        dev1 = torch.device("mlu:1")

        # Unfortunately I need to make the tensors largeish.
        # Bigger tensors = longer D2D transfers = more likely to expose races.
        size = 2**26

        a = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)
        b = torch.full((size,), 1, device=dev1, dtype=torch.float64, requires_grad=True)

        # Here to_backward_recipient = a*b is used only once, so MulBackward's InputBuffer slot only expects 1 input.
        # This tests the situation where we don't call InputBuffer::accumulate for MulBackward's InputBuffer.
        to_backward_recipient = a * b
        s = to_backward_recipient.to(device="mlu:0").sum()
        torch.mlu.synchronize(device=dev0)
        torch.mlu.synchronize(device=dev1)
        s.backward()
        self.assertTrue(a.grad.sum().item() == size)
        self.assertTrue(b.grad.sum().item() == size)

        # Here to_backward_recipient = a*b is used twice, so MulBackward's InputBuffer slot expects 2 inputs.
        # This tests the situation where we do call InputBuffer::accumulate for MulBackward's InputBuffer.
        a.grad = None
        b.grad = None
        to_backward_recipient = a * b
        # Multiply by 2 here so to's backward creates gradient values that are different from the case above,
        # to mitigate weirdness if the caching allocator happens to reuse memory regions that were populated
        # with 1s by the case above
        s0 = to_backward_recipient.to(device="mlu:0").sum() * 2.0
        s1 = to_backward_recipient.to(device="mlu:0").sum() * 2.0
        torch.mlu.synchronize(device=dev0)
        torch.mlu.synchronize(device=dev1)
        s0.backward(retain_graph=True)
        s1.backward()
        self.assertTrue(a.grad.sum().item() == 4 * size)
        self.assertTrue(b.grad.sum().item() == 4 * size)

    # @unittest.skip("not test")
    @testinfo()
    def test_record_stream(self):
        # cycles_per_ms = get_cycles_per_ms()

        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        result = torch.mlu.FloatTensor(t.size())
        stream = torch.mlu.Stream()
        ptr = [None]
        x = torch.zeros((1024, 10240), dtype=torch.float32, device="mlu")
        y = torch.zeros((10240, 1024), dtype=torch.float32, device="mlu")

        # Performs the CPU->MLU copy in a background stream
        def perform_copy():
            with torch.mlu.stream(stream):
                tmp = t.mlu(non_blocking=True)
                ptr[0] = tmp.data_ptr()
            torch.mlu.current_stream().wait_stream(stream)
            tmp.record_stream(torch.mlu.current_stream())
            # torch.mlu._sleep(int(50 * cycles_per_ms))  # delay the copy
            # torch.matmul instead of torch.mlu._sleep
            for i in range(1, 20):
                torch.matmul(x, y)
            result.copy_(tmp)

        perform_copy()
        with torch.mlu.stream(stream):
            tmp2 = torch.mlu.FloatTensor(t.size())
            tmp2.zero_()
            self.assertNotEqual(
                tmp2.data_ptr(), ptr[0], msg="allocation re-used to soon"
            )

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        # Check that the block will be re-used after the main stream finishes
        torch.mlu.current_stream().synchronize()
        with torch.mlu.stream(stream):
            tmp3 = torch.mlu.FloatTensor(t.size())
            self.assertEqual(tmp3.data_ptr(), ptr[0], msg="allocation not re-used")

    # @unittest.skip("not test")
    @testinfo()
    def test_caching_allocator_record_stream_oom(self):
        """allocations delayed by a record_stream call should still be freed on
        an out-of-memory in mlu_malloc_retry. see issue #19219"""
        stream = torch.mlu.Stream()

        with torch.mlu.stream(stream):
            y = torch.zeros(4 * 1024 * 1024, device="mlu")

        for _ in range(100):
            x = torch.empty(4 * 1024 * 1024, device="mlu")
            with torch.mlu.stream(stream):
                y += x
            # delays re-use of `x` until after all operations in `stream`
            x.record_stream(stream)
            del x

        # we've made a mess by allocating up to the device capacity. free any
        # cached blocks in case it affects future tests.
        torch.mlu.empty_cache()

    @contextlib.contextmanager
    def _get_external_stream(self, device):
        with device:
            yield torch.mlu.Stream().mlu_stream

    # @unittest.skip("not test")
    @testinfo()
    def test_external_streams(self):
        device = torch.mlu.device(0)
        with self._get_external_stream(device) as stream_v:
            ext_stream = torch.mlu.ExternalStream(stream_v)
            self.assertEqual(stream_v, ext_stream.mlu_stream)
            self.assertEqual(ext_stream.device.index, device.idx)

    # @unittest.skip("not test")
    @testinfo()
    def test_external_streams_multi_device(self):
        if torch.mlu.device_count() <= 1:
            return
        device = torch.mlu.device(1)
        with self._get_external_stream(device) as stream_v:
            ext_stream = torch.mlu.ExternalStream(stream_v, device=device)
            self.assertEqual(stream_v, ext_stream.mlu_stream)
            self.assertEqual(ext_stream.device.index, device.idx)

    # @unittest.skip("not test")
    @testinfo()
    def test_multi_stream_same_device(self):
        s0 = torch.mlu.Stream()
        s1 = torch.mlu.Stream()
        self.assertEqual(s0.device, s1.device)
        self.assertNotEqual(s0.mlu_stream, s1.mlu_stream)

    # @unittest.skip("not test")
    @testinfo()
    def test_avoid_driver_default_queue(self):
        if torch.mlu.device_count() <= 1:
            return
        default_stream_d0 = torch.mlu.current_stream()
        default_stream_d1 = torch.mlu.current_stream(device=1)
        self.assertNotEqual(default_stream_d0.mlu_stream, 0)
        self.assertNotEqual(default_stream_d1.mlu_stream, 0)


if __name__ == "__main__":
    unittest.main()
