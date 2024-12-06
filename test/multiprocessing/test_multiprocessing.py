import os
import gc
import sys
import time
import copy
import logging
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import bytes_to_scalar
import numpy as np
import torch.multiprocessing as mp
from torch.nn import Parameter

import unittest  # pylint: disable=C0411

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

MAX_WAITING_TIME_IN_SECONDS = 30
HAS_SHM_FILES = os.path.isdir("/dev/shm")
TEST_MULTIMLU = torch.mlu.device_count() > 1


class leak_checker:
    def __init__(self, test_case):
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        self.next_fds = self._get_next_fds(10)
        return self

    def __exit__(self, *args):
        if torch.mlu.is_available():
            torch.mlu.ipc_collect()
        if args[0] is None:
            self.test_case.assertFalse(self.has_shm_files())
        return False

    def check_pid(self, pid):
        self.checked_pids.append(pid)

    def _get_next_fds(self, n=1):
        # dup uses the lowest-numbered unused descriptor for the new descriptor
        fds = [os.dup(0) for i in range(n)]
        for fd in fds:
            os.close(fd)
        return fds

    def has_shm_files(self, wait=True):
        if not HAS_SHM_FILES:
            return False

        result = self._has_shm_files()
        if not result or mp.get_sharing_strategy() != "file_system" or not wait:
            return result

        total_waiting_time = 0
        waiting_time = 0.5

        while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and result:
            time.sleep(waiting_time)
            total_waiting_time += waiting_time
            result = self._has_shm_files()

        return result

    def _has_shm_files(self):
        gc.collect()
        names = ["torch_" + str(pid) for pid in self.checked_pids]
        for filename in os.listdir("/dev/shm"):
            for name in names:
                if filename.startswith(name):
                    return True
        return False


def integer_parameter_serialization(iparam):
    result = iparam + 3
    return result


def autograd_sharing(queue, ready, master_modified, device, is_parameter):
    var = queue.get()
    ready.set()
    master_modified.wait()

    expected_var = torch.arange(1.0, 26, device=device).view(5, 5)
    expected_var[0, 0] = 1000
    is_ok = var.data.equal(expected_var)
    var.data[:] = torch.ones(5, 5, device=device)

    is_ok &= var.grad is None
    is_ok &= not var._backward_hooks
    if is_parameter:
        is_ok &= type(var) == Parameter
    else:
        is_ok &= type(var) == torch.Tensor
    var._grad = torch.ones(5, 5, device=device)

    queue.put(is_ok)


def mixed_type_producer(queue, event):
    for _ in range(10):
        float_tensor = torch.ones(16, 1024, 1024).float().mlu()
        byte_tensor = torch.zeros(1024, 1024).byte().mlu()
        queue.put(float_tensor)
        queue.put(byte_tensor)
        event.wait()
        event.clear()


def ipc_producer(queue, event):
    float_tensor = torch.ones(16, 16).float().mlu()
    for _ in range(10):
        queue.put(float_tensor)
        event.wait()
        event.clear()


def requires_grad_variable_sharing(queue, ready):
    var = queue.get()
    ready.set()
    queue.put(var.requires_grad)


def sum_tensors(inq, outq):
    with torch.mlu.device(1):
        tensors = inq.get()
        for tensor in tensors:
            outq.put(
                (
                    tensor.sum(),
                    tensor.get_device(),
                    tensor.numel(),
                    tensor.storage().size(),
                )
            )


def simple_fill(queue, event):
    data = queue.get()
    data[0][:] = 4
    event.set()


def send_tensor(queue, event, device, dtype):
    t = torch.ones(5, 5, device=device, dtype=dtype)
    queue.put(t)
    queue.put(t)
    event.wait()


def send_and_delete_tensors(queue, event, device, dtype, count, size=5):
    for i in range(count):
        t = torch.full([size], i, device=device, dtype=dtype)
        queue.put(t)
        del t
    event.wait()


def _test_mlu_ipc_deadlock_actor(queue, iterations):
    for i in range(iterations):
        if not queue.empty():
            queue.get()
        time.sleep(0.01)


def _test_mlu_ipc_deadlock_learner(queue, iterations):
    net = torch.nn.LSTM(1, 1).mlu()
    for i in range(iterations):
        if not queue.full():
            queue.put(copy.deepcopy(net.state_dict()))
        time.sleep(0.01)


def receive_and_send_sum(queue, out_queue, event, device, dtype, count, size=5):
    s = torch.full([size], 0, device=device, dtype=dtype)
    for i in range(count):
        t = queue.get()
        s += t
    out_queue.put(s)
    event.wait()


def receive_and_send(queue, out_queue, event, count):
    for i in range(count):
        t = queue.get()
        out_queue.put(t.clone())
    event.wait()


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


def temporarily_replace__sleep():
    x = torch.zeros((1024, 10240), dtype=torch.float32, device="mlu")
    y = torch.zeros((10240, 1024), dtype=torch.float32, device="mlu")
    for i in range(1000):
        torch.matmul(x, y)


def _test_sub_map(q1, e1, q2):
    for i in range(10):
        ll = []
        while True:
            e1.wait()
            e1.clear()
            shape = q1.get(block=True)
            if shape is None:
                break
            t1 = torch.randn(shape, dtype=torch.float32).mlu()
            ll.append(t1)
            t1.mul_(2)
            q2.put(t1)
        del ll
        torch.mlu.empty_cache()


class Multiprocessing(TestCase):
    def _test_sharing(self, ctx=mp, device="mlu", dtype=torch.float, repeat=1):
        def test_fill():
            x = torch.zeros(5, 5).to(device, dtype)
            q = ctx.Queue()
            e = ctx.Event()

            data = [x, x[:, 1]]
            q.put(data)

            p = ctx.Process(target=simple_fill, args=(q, e))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            total_waiting_time = 0
            waiting_time = 0.5
            is_set = False
            # Once the child process is done, it will set the event to notify the
            # parent accordingly
            while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and not is_set:
                time.sleep(waiting_time)
                total_waiting_time += waiting_time
                is_set = e.is_set()

            self.assertTrue(is_set)
            self.assertTrue(data[0].eq(4).all())
            self.assertTrue(data[1].eq(4).all())

            p.join(100)
            self.assertFalse(p.is_alive())

        def test_receive():
            q = ctx.Queue()
            e = ctx.Event()

            p = ctx.Process(target=send_tensor, args=(q, e, device, dtype))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            t1 = q.get()
            t2 = q.get()
            self.assertTrue(t1.eq(1).all())
            s1 = t1.storage()
            s2 = t2.storage()
            self.assertEqual(type(s1), type(s2))
            self.assertEqual(s1.data_ptr(), s1.data_ptr())
            self.assertEqual(s1, s2)

            # We need to delete this tensors to allow producer (child process)
            # collect them properly
            del t1, t2

            # Mark the event as done and join the process
            e.set()
            p.join(100)
            self.assertFalse(p.is_alive())

        with leak_checker(self) as lc:
            for _ in range(repeat):
                test_fill()
                test_receive()

    # @unittest.skip("not test")
    @testinfo()
    def test_integer_parameter_serialization_mlu(self):
        device = "mlu"
        param = Parameter(
            torch.tensor(0, dtype=torch.float32).mlu(), requires_grad=False
        )
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=integer_parameter_serialization, args=(param,))
        p.start()
        p.join()
        self.assertEqual(
            0,
            p.exitcode,
            msg=f'Failed to serialize successfully for "{device}" device!',
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_integer_variable_serialization_mlu(self):
        device = "mlu"
        param = torch.tensor(0, dtype=torch.float32, device=device, requires_grad=True)
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=integer_parameter_serialization, args=(param,))
        p.start()
        p.join()
        self.assertEqual(
            0,
            p.exitcode,
            msg=f'Failed to serialize successfully for "{device}" device!',
        )

    def _test_autograd_sharing(self, var, ctx=mp, is_parameter=False):
        device = "mlu"
        ready = ctx.Event()
        master_modified = ctx.Event()
        queue = ctx.Queue()
        p = ctx.Process(
            target=autograd_sharing,
            args=(queue, ready, master_modified, device, is_parameter),
        )
        p.daemon = True
        p.start()

        # This would cause an error if we tried to serialize the hooks,
        # because it's a closure and pickle doesn't support closures.
        @torch.utils.hooks.unserializable_hook
        def hook(*unused):
            pass

        if var.requires_grad:
            var.register_hook(hook)
        var._grad = torch.zeros(5, 5, device=device)
        queue.put(var)
        ready.wait()
        var.data[0, 0] = 1000
        var.grad.data[:] = torch.ones(5, 5, device=device) * 4
        master_modified.set()
        worker_ok = queue.get()
        self.assertTrue(worker_ok)
        self.assertEqual(var.data, torch.ones(5, 5, device=device))
        self.assertEqual(var.grad.data, torch.ones(5, 5, device=device) * 4)
        p.join(100)
        self.assertFalse(p.is_alive())
        torch.mlu.ipc_collect()

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_parameter_sharing(self):
        param = Parameter(torch.arange(1.0, 26, device="mlu").view(5, 5))
        self._test_autograd_sharing(param, mp.get_context("spawn"), is_parameter=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_variable_sharing(self):
        for requires_grad in [True, False]:
            var = (
                torch.arange(1.0, 26, device="mlu")
                .view(5, 5)
                .requires_grad_(requires_grad)
            )
            self._test_autograd_sharing(var, mp.get_context("spawn"))

    # @unittest.skip("not test")
    @testinfo()
    def test_mixed_types_mlu_sharing(self):
        ctx = mp.get_context("spawn")
        all_ones = torch.ones(16, 1024, 1024).float()
        all_zeros = torch.zeros(1024, 1024).byte()
        queue = ctx.Queue()
        event = ctx.Event()

        p = ctx.Process(target=mixed_type_producer, args=(queue, event))

        p.start()

        for _ in range(10):
            float_tensor = queue.get()
            byte_tensor = queue.get()

            self.assertEqual(float_tensor, all_ones)
            self.assertEqual(byte_tensor, all_zeros)
            del float_tensor, byte_tensor
            event.set()

        time.sleep(5)
        p.join()
        torch.mlu.ipc_collect()

    # @unittest.skip("not test")
    @testinfo()
    def test_release_ipc_counter_mlu(self):
        ctx = mp.get_context("spawn")
        all_ones = torch.ones(16, 16).float()
        queue = ctx.Queue()
        event = ctx.Event()
        p = ctx.Process(target=ipc_producer, args=(queue, event))
        p.start()
        for _ in range(10):
            float_tensor = queue.get()
            self.assertEqual(float_tensor, all_ones)
            del float_tensor
            event.set()
        time.sleep(5)
        p.join()
        torch.mlu.ipc_collect()

    # @unittest.skip("not test")
    @testinfo()
    def test_leaf_variable_sharing(self):
        device = "mlu"
        for requires_grad in [True, False]:
            var = (
                torch.arange(1.0, 26, device=device)
                .view(5, 5)
                .requires_grad_(requires_grad)
            )
            self.assertTrue(var.is_leaf)
            ctx = mp.get_context("spawn")
            ready = ctx.Event()
            queue = ctx.Queue()
            p = ctx.Process(target=requires_grad_variable_sharing, args=(queue, ready))
            p.daemon = True
            p.start()
            queue.put(var)
            ready.wait()
            worker_requires_grad = queue.get()
            self.assertTrue(worker_requires_grad == requires_grad)
        torch.mlu.ipc_collect()

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_simple(self):
        torch.mlu.FloatTensor([1])  # initialize MLU outside of leak checker
        self._test_sharing(mp.get_context("spawn"), "mlu", torch.float)

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_memory_allocation(self):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        e = ctx.Event()
        p = ctx.Process(
            target=send_and_delete_tensors, args=(q, e, "mlu", torch.int, 5)
        )
        p.start()
        t = []
        for _ in range(5):
            t.append(q.get())
        self.assertEqual(t[0], torch.full([5], 0, dtype=torch.int32))
        del t
        e.set()
        p.join(1)

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_ipc_deadlock(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(1)
        processes = dict(
            a=ctx.Process(target=_test_mlu_ipc_deadlock_actor, args=(queue, 100)),
            l=ctx.Process(target=_test_mlu_ipc_deadlock_learner, args=(queue, 100)),
        )

        for p in processes.values():
            p.start()

        for p in processes.values():
            # TODO(PYTORCH-12191): Temporarily increase the time limit until common_utils.py is optimized
            p.join(30)

        for p in processes.values():
            self.assertFalse(p.is_alive())

    # @unittest.skip("not test")
    @unittest.skipIf(not TEST_MULTIMLU, "found only 1 MLU")
    @testinfo()
    def test_mlu_small_tensors(self):
        # Check multiple small tensors which will likely use the same
        # underlying cached allocation
        ctx = mp.get_context("spawn")
        tensors = []
        for i in range(5):
            device = i % 2
            tensors += [torch.arange(i * 5.0, (i + 1) * 5).mlu(device)]

        inq = ctx.Queue()
        outq = ctx.Queue()
        inq.put(tensors)
        p = ctx.Process(target=sum_tensors, args=(inq, outq))
        p.start()

        results = []
        for _ in range(5):
            results.append(outq.get())
        p.join()

        for i, _tensor in enumerate(tensors):
            v, device, tensor_size, storage_size = results[i]
            self.assertEqual(v, torch.arange(i * 5.0, (i + 1) * 5).sum())
            self.assertEqual(device, i % 2)
            self.assertEqual(tensor_size, 5)

            # You might think this should be the case, but it's not!  After
            # data from the MLU caching allocator goes through IPC, the
            # size of the storage is the size of the *cached cudaMalloc for
            # the entire memory block* of the storage, not just the storage.
            # See Note [MLU IPC and the caching allocator] for more info
            #
            # self.assertEqual(storage_size, 5)

        # Collect current process (producer) files, make sure nothing holds
        # ref to the sent tensors
        del _tensor
        del tensors

        torch.mlu.ipc_collect()

    # @unittest.skip("not test")
    @testinfo()
    def test_mlu_send_many(self, name=None, size=5, count=1000):
        ctx = mp.get_context("spawn")
        q1 = ctx.Queue()
        q2 = ctx.Queue()
        q3 = ctx.Queue()
        e1 = ctx.Event()
        e2 = ctx.Event()
        e3 = ctx.Event()
        p1 = ctx.Process(
            target=send_and_delete_tensors,
            args=(q1, e1, "mlu", torch.float, count, size),
        )
        p2 = ctx.Process(target=receive_and_send, args=(q1, q2, e2, count))
        p3 = ctx.Process(
            target=receive_and_send_sum,
            args=(q2, q3, e3, "mlu", torch.float, count, size),
        )
        p1.start()
        p2.start()
        p3.start()
        result = q3.get()
        self.assertEqual(result[0], int(count * (count - 1) / 2))
        del result
        e1.set()
        e2.set()
        e3.set()
        p1.join(1)
        p2.join(1)
        p3.join(1)

    # @unittest.skip("not test")
    @testinfo()
    def test_event(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        ready = ctx.Event()
        done = ctx.Event()
        p = ctx.Process(target=mlu_multiply_two, args=(queue, ready, done))
        p.start()

        ready.wait()
        with torch.mlu.stream(torch.mlu.Stream()):
            tensor = torch.mlu.FloatTensor([1, 1, 1, 1])
            event = torch.mlu.Event(interprocess=True)
            tensor.add_(1)
            event.record()
            queue.put((event, tensor))
            done.wait()  # must wait until subprocess records event
            event.synchronize()
            self.assertEqual(list(tensor), [4, 4, 4, 4])
        p.join()

    @staticmethod
    def _test_event_multiprocess_child(event, p2c, c2p):
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for record in parent
        event.synchronize()
        c2p.put(1)  # notify parent synchronization is done

    # @unittest.skip("not test")
    @testinfo()
    def test_event_multiprocess(self):
        event = torch.mlu.Event(enable_timing=False, interprocess=True)
        self.assertTrue(event.query())

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=Multiprocessing._test_event_multiprocess_child,
            args=(event, p2c, c2p),
        )
        p.start()
        c2p.get()  # wait for until child process is ready
        # Now mlu not support _sleep ops.
        # torch.matmul instead of torch.mlu._sleep
        # torch.mlu._sleep(50000000)  # spin for about 50 ms
        temporarily_replace__sleep()

        event.record()
        p2c.put(0)  # notify child event is recorded
        self.assertFalse(event.query())
        c2p.get()  # wait for synchronization in child
        self.assertTrue(event.query())
        p.join()

    # @unittest.skip("not test")
    @unittest.skipIf(not TEST_MULTIMLU, "found only 1 MLU")
    @testinfo()
    def test_event_handle_multi_mlu(self):
        d0 = torch.device("mlu:0")
        d1 = torch.device("mlu:1")
        with torch.mlu.device(d0):
            e0 = torch.mlu.Event(enable_timing=False, interprocess=True)

        with torch.mlu.device(d1):
            # create handle on different device from un-recorded event
            e0.ipc_handle()

        with torch.mlu.device(d0):
            e1 = torch.mlu.Event(enable_timing=False, interprocess=True)
            stream = torch.mlu.Stream()
            # Now mlu not support _sleep ops.
            # torch.matmul instead of torch.mlu._sleep
            # torch.mlu._sleep(50000000)  # spin for about 50 ms
            temporarily_replace__sleep()
            e1.record(stream)

        with torch.mlu.device(d1):
            # create handle on different device from recorded event
            e1.ipc_handle()

    @staticmethod
    def _test_event_handle_importer_consumer(handle, p2c, c2p):
        e1 = torch.mlu.Event.from_ipc_handle(0, handle)
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for record in parent
        e1.synchronize()
        c2p.put(1)  # notify synchronization is done in child
        p2c.get()  # wait for parent to finish before destructing child event

    # @unittest.skip("not test")
    @testinfo()
    def test_event_handle_importer(self):
        e0 = torch.mlu.Event(enable_timing=False, interprocess=True)
        self.assertTrue(e0.query())

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=Multiprocessing._test_event_handle_importer_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for child to become ready
        # Now mlu not support _sleep ops.
        # torch.matmul instead of torch.mlu._sleep
        # torch.mlu._sleep(50000000)  # spin for about 50 ms
        temporarily_replace__sleep()
        e0.record()
        p2c.put(0)  # notify child event is recorded

        self.assertFalse(e0.query())
        c2p.get()  # wait for synchronization in child
        self.assertTrue(e0.query())
        p2c.put(1)  # notify child that parent is done
        p.join()

    @staticmethod
    def _test_event_handle_exporter_consumer(handle, p2c, c2p):
        stream = torch.mlu.Stream()
        with torch.mlu.stream(stream):
            e1 = torch.mlu.Event.from_ipc_handle(torch.mlu.current_device(), handle)
            # Now mlu not support _sleep ops.
            # torch.matmul instead of torch.mlu._sleep
            # torch.mlu._sleep(50000000)  # spin for about 50 ms
            temporarily_replace__sleep()
            e1.record()
            c2p.put(0)
            # wait for parent process finished synchronization before
            # destructing e1
            p2c.get()

    # @unittest.skip("not test")
    @testinfo()
    def test_event_handle_exporter(self):
        e0 = torch.mlu.Event(enable_timing=False, interprocess=True)

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=Multiprocessing._test_event_handle_exporter_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()
        # wait for event in child process is recorded
        c2p.get()

        self.assertFalse(e0.query())
        e0.synchronize()
        self.assertTrue(e0.query())
        p2c.put(0)
        p.join()

    # See Note [ipc handle ptrmap]
    # Here, it is simulated that after a piece of memory ptr_a is cnrtFreed, when the
    # memory address applied again is the same as ptr_a, the internal ptrmap_ipc can
    # update the key-value pair of {ptr, handle}. The order of the test is:

    # 1) The tensor performs ipc communication and is stored in the list After the first for loop,
    # the tensor memory is released uniformly. At this time, the internal ptrmap_ipc should sense
    # that ptr is released and delete the key-value pair of {ptr, handle}

    # 2) In the second for loop, the tensor creation will apply for the same memory address. At this time,
    # because the previous key-value pair of ptr has been deleted in ptrmap_ipc, it is necessary to
    # call cnrtAcquireMemHandle again to obtain the correct handle when performing ipc.
    # @unittest.skip("not test")
    @testinfo()
    def test_ipc_with_workround(self):
        ctx = mp.get_context("spawn")
        q1 = ctx.Queue(maxsize=1)
        q2 = ctx.Queue(maxsize=1)
        e1 = ctx.Event()
        p1 = ctx.Process(
            target=_test_sub_map,
            args=(q1, e1, q2),
        )
        p1.start()
        for i in range(10):
            shape = (32, 1024, 1024)
            num_iterations = 10
            l = []
            for j in range(num_iterations):
                q1.put(shape, block=True)
                e1.set()
                l.append(q2.get(block=True))
            del l
            q1.put(None)
            e1.set()
        p1.join()


if __name__ == "__main__":
    unittest.main()
