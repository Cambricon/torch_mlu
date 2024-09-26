import unittest
import sys
import os
import numpy as np
import subprocess
import ctypes

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

import torch
import torch.nn as nn
import torch_mlu
import torch.distributed as dist


class AbstractTestCases(TestCase):
    # @unittest.skip("not test")
    def test_tesnor_device(self):
        def assertEqual(device_str, fn):
            self.assertEqual(torch.device(device_str), fn().device)
            self.assertEqual(device_str, str(fn().device))

        self.assertEqual(torch.mlu.is_available(), True)

        if torch.mlu.is_available():
            assertEqual("mlu:0", lambda: torch.tensor(5).mlu(0))
            assertEqual("mlu:0", lambda: torch.tensor(5).mlu("mlu:0"))
            self.assertRaises(RuntimeError, lambda: torch.tensor(5).mlu("cpu"))
            self.assertRaises(RuntimeError, lambda: torch.tensor(5).mlu("cpu:0"))
            assertEqual(
                "mlu:0", lambda: torch.tensor(5, dtype=torch.int64, device="mlu:0")
            )
            assertEqual(
                "mlu:" + str(torch.mlu.current_device()),
                lambda: torch.tensor(5, dtype=torch.int64, device="mlu"),
            )
            assertEqual(
                "mlu:0", lambda: torch.tensor(np.random.randn(2, 3), device="mlu:0")
            )

    # @unittest.skip("not test")
    def test_to(self):
        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(
                t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True)
            )

            devices = [t.device]
            if t.device.type == "mlu":
                if t.device.index == -1:
                    devices.append("mlu:{}".format(torch.mlu.current_device()))
                elif t.device.index == torch.mlu.current_device():
                    devices.append("mlu")
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(
                    t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True)
                )

        a = torch.tensor(5)
        for non_blocking in [True, False]:
            for mlu in ["mlu", "mlu:0"]:
                b = torch.tensor(5.0, device=mlu)
                test_copy_behavior(b, non_blocking)
                self.assertEqual(b.device, b.to(mlu, non_blocking=non_blocking).device)
                self.assertEqual(
                    a.device, b.to("cpu", non_blocking=non_blocking).device
                )
                self.assertEqual(b.device, a.to(mlu, non_blocking=non_blocking).device)
                self.assertIs(
                    torch.int32,
                    b.to("cpu", dtype=torch.int32, non_blocking=non_blocking).dtype,
                )
                self.assertEqual(
                    a.device,
                    b.to("cpu", dtype=torch.int32, non_blocking=non_blocking).device,
                )
                self.assertIs(torch.int32, b.to(dtype=torch.int32).dtype)
                self.assertEqual(b.device, b.to(dtype=torch.int32).device)

    # @unittest.skip("not test")
    def test_mlu_tensor(self):
        default_type = torch.Tensor().type()
        tensor_classes = [
            # torch.mlu.DoubleTensor,
            torch.mlu.FloatTensor,
            torch.mlu.HalfTensor,
            torch.mlu.LongTensor,
            torch.mlu.IntTensor,
            torch.mlu.ShortTensor,
            torch.mlu.CharTensor,
            torch.mlu.BoolTensor,
            torch.mlu.ByteTensor,
        ]
        tmp = np.random.rand(10, 10)
        for t in tensor_classes:
            # new(*, Device? device=None)
            obj0 = t(device=0)
            obj0.__repr__()
            str(obj0)
            self.assertTrue(isinstance(obj0, t))

            # new(IntArrayRef size, *, Device? device=None)
            obj4 = t((10, 10), device=1).fill_(1)
            obj4.__repr__()
            str(obj4)
            self.assertTrue(isinstance(obj4, t))

            # new(PyObject* data, *, Device? device=None)
            obj5 = t(tmp).fill_(1)
            obj5.__repr__()
            str(obj5)
            self.assertTrue(isinstance(obj5, t))

            # new(Tensor other)
            obj3 = t(obj4)
            obj4.__repr__()
            str(obj4)
            self.assertTrue(isinstance(obj4, t))

            # TODO(dengshun): support new(Storage storage) after mlu storage is supported

            # FloatTensor with type
            obj_type = torch.rand(2, 3).type(t, non_blocking=True)
            obj_type.__repr__()
            str(obj_type)
            self.assertTrue(isinstance(obj_type, t))

            # FloatTensor functions
            obj_func_mlu = torch.mlu.FloatTensor(
                [2, 3], device=1
            ) + torch.mlu.FloatTensor([2, 3], device=1)
            obj_func_cpu = torch.FloatTensor([2, 3]) + torch.FloatTensor([2, 3])
            self.assertEqual(obj_func_mlu.cpu(), obj_func_cpu)

            # FloatTensor wrap by device
            with torch.mlu.device(1):
                obj_wrap = torch.mlu.FloatTensor([2, 3])
            self.assertEqual(obj_wrap.device.index, 1)

    # @unittest.skip("not test")
    def test_mlu_tensor_ismlu(self):
        if torch.mlu.is_available():
            self.assertTrue(torch.mlu.IntTensor.is_mlu)
            self.assertTrue(torch.mlu.HalfTensor.is_mlu)
            self.assertTrue(torch.mlu.FloatTensor.is_mlu)
            self.assertTrue(torch.mlu.DoubleTensor.is_mlu)
            self.assertTrue(torch.mlu.ByteTensor.is_mlu)
            self.assertTrue(torch.mlu.LongTensor.is_mlu)

    # @unittest.skip("not test")
    def test_mlu_tensor_type_ismlu(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.mlu.is_available():
            self.assertIsInstance(x.float().mlu(), torch.mlu.FloatTensor)
            self.assertIsInstance(x.int().mlu(), torch.mlu.IntTensor)
            self.assertIsInstance(x.int().mlu().cpu(), torch.IntTensor)
            self.assertTrue(x.float().mlu().is_mlu)
            self.assertTrue(x.int().mlu().is_mlu)

            self.assertIsInstance(x.mlu().double(), torch.mlu.DoubleTensor)
            self.assertIsInstance(x.mlu().float(), torch.mlu.FloatTensor)
            self.assertIsInstance(x.mlu().float().cpu(), torch.FloatTensor)
            self.assertIsInstance(x.mlu().float().cpu().int(), torch.IntTensor)
            self.assertTrue(x.mlu().double().is_mlu)
            self.assertTrue(x.mlu().float().is_mlu)

    # @unittest.skip("not test")
    def test_to(self):
        m = nn.Linear(3, 5)
        self.assertIs(m, m.to("cpu"))
        self.assertIs(m, m.to("cpu", dtype=torch.float32))
        self.assertEqual(m.half(), m.to(torch.float16))
        self.assertRaises(RuntimeError, lambda: m.to("cpu", copy=True))

        if torch.mlu.is_available():
            for mlu in ["mlu", "mlu:0"]:
                m2 = m.mlu(device=mlu)
                self.assertIs(m2, m2.to(mlu))
                self.assertEqual(m, m2.to("cpu"))
                self.assertEqual(m2, m.to(mlu))
                self.assertIs(m2, m2.to(dtype=torch.float32))
                self.assertEqual(m2.half(), m2.to(dtype=torch.float16))

    # @unittest.skip("not test")
    def test_automlu(self):
        x = torch.randn(5, 5).mlu()
        y = torch.randn(5, 5).mlu()
        self.assertEqual(x.get_device(), 0)
        self.assertEqual(x.get_device(), 0)
        with torch.mlu.device(1):
            z = torch.randn(5, 5).mlu()
            self.assertEqual(z.get_device(), 1)
            q = x.add(y)
            self.assertEqual(q.get_device(), 0)
            w = torch.randn(5, 5).mlu()
            self.assertEqual(w.get_device(), 1)
            self.assertEqual(y.mlu().get_device(), 1)
        z = z.mlu()
        self.assertEqual(z.get_device(), 0)

    # @unittest.skip("not test")
    def test_copy_device(self):
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
    def test_sum_fp16(self):
        x = torch.zeros(10, device="mlu", dtype=torch.float16)
        self.assertEqual(x.sum(), 0)

        # x = torch.ones(65504, device='mlu', dtype=torch.float16)
        # self.assertEqual(x.sum(), 65504)
        # self.assertEqual(x.sum(dtype=torch.float32), 65504)

        # x = torch.ones(65536, device='mlu', dtype=torch.float16)
        # self.assertEqual(x.sum(dtype=torch.float32), 65536)

        a = torch.zeros(1203611).bernoulli_(0.0005)
        x = a.to(device="mlu", dtype=torch.float16)
        self.assertEqual(x.sum().item(), a.sum().item())

        a = torch.zeros(100, 121, 80).bernoulli_(0.0005)
        x = a.to(device="mlu", dtype=torch.float16)
        self.assertEqual(x.sum((0, 2)).float().cpu(), a.sum((0, 2)))

    # @unittest.skip("not test")
    def test_mean_fp16(self):
        x = torch.ones(65536, device="mlu", dtype=torch.float16)
        self.assertEqual(x.mean(), 1)

        # x = torch.ones(65536, device='mlu', dtype=torch.float16)
        # self.assertEqual(x.mean(dtype=torch.float32), 1)

    # @unittest.skip("not test")
    def test_mlu_set_device(self):
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
    def test_mlu_synchronize(self):
        torch.mlu.synchronize()
        torch.mlu.synchronize("mlu")
        torch.mlu.synchronize("mlu:0")
        torch.mlu.synchronize(0)
        torch.mlu.synchronize(torch.device("mlu:0"))

        torch.mlu.synchronize("mlu:1")
        torch.mlu.synchronize(1)
        torch.mlu.synchronize(torch.device("mlu:1"))

        with self.assertRaisesRegex(ValueError, "Expected a mlu device, but got: cpu"):
            torch.mlu.synchronize(torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "Expected a mlu device, but got: cpu"):
            torch.mlu.synchronize("cpu")

    # @unittest.skip("not test")
    def test_current_stream(self):
        d0 = torch.device("mlu:0")
        d1 = torch.device("mlu:1")

        s0 = torch.mlu.current_stream()
        s1 = torch.mlu.current_stream(device=1)
        s2 = torch.mlu.current_stream(device=0)

        self.assertEqual(0, s0.device.index)
        self.assertEqual(1, s1.device.index)
        self.assertEqual(0, s2.device.index)
        self.assertEqual(s0, s2)

        with torch.mlu.device(d1):
            s0 = torch.mlu.current_stream()
            s1 = torch.mlu.current_stream(1)
            s2 = torch.mlu.current_stream(d0)

        self.assertEqual(1, s0.device.index)
        self.assertEqual(1, s1.device.index)
        self.assertEqual(0, s2.device.index)
        self.assertEqual(s0, s1)

    # @unittest.skip("not test")
    def test_default_stream(self):
        d0 = torch.device("mlu:0")
        d1 = torch.device("mlu:1")

        with torch.mlu.device(d0):
            s0 = torch.mlu.default_stream()

        with torch.mlu.device(d1):
            s1 = torch.mlu.default_stream()

        s2 = torch.mlu.default_stream(device=0)
        s3 = torch.mlu.default_stream(d1)

        self.assertEqual(0, s0.device.index)
        self.assertEqual(1, s1.device.index)
        self.assertEqual(0, s2.device.index)
        self.assertEqual(1, s3.device.index)
        self.assertEqual(s0, s2)
        self.assertEqual(s1, s3)

        with torch.mlu.device(d0):
            self.assertEqual(torch.mlu.current_stream().mlu_stream, s0.mlu_stream)

        with torch.mlu.device(d1):
            self.assertEqual(torch.mlu.current_stream().mlu_stream, s1.mlu_stream)

        with self.assertRaisesRegex(ValueError, "Expected a mlu device, but got: cpu"):
            torch.mlu.default_stream(torch.device("cpu"))

    @unittest.skip("not test")
    def test_randn(self):
        dtype = torch.float
        for device in ["mlu", "mlu:0", torch.device("mlu"), torch.device("mlu:0")]:
            for size in [0, 100]:
                torch.manual_seed(123456)
                res1 = torch.randn(size, size, dtype=dtype, device=device)
                torch.manual_seed(123456)
                res2 = torch.randn(size, size)
                self.assertEqual(res1, res2)

    # @unittest.skip("not test")
    def test_zeros(self):
        boolTensor = torch.zeros(2, 2, dtype=torch.bool, device="mlu")
        expected = torch.tensor([[False, False], [False, False]], dtype=torch.bool)
        self.assertEqual(boolTensor, expected)

        halfTensor = torch.zeros(1, 1, dtype=torch.half, device="mlu")
        expected = torch.tensor([[0.0]], dtype=torch.float16)
        self.assertEqual(halfTensor, expected)

    # @unittest.skip("not test")
    def test_linspace(self):
        floatTensor = torch.linspace(1, 10, steps=2, dtype=torch.float32, device="mlu")
        expected = torch.linspace(1, 10, steps=2, dtype=torch.float32)
        self.assertEqual(floatTensor, expected)

    # @unittest.skip("not test")
    def test_get_properties(self):
        c1 = torch.mlu.get_device_properties(0)
        c2 = torch.mlu.get_device_properties("mlu:0")
        c3 = torch.mlu.get_device_properties(torch.device("mlu:0"))

        self.assertEqual(c1, c2)
        self.assertEqual(c1, c3)

    # @unittest.skip("not test")
    def test_mlu_load(self):
        m = nn.Linear(3, 5).mlu()
        torch.save(m, "m.pth")
        m_load = torch.load("m.pth", map_location="mlu")
        self.assertEqual(m_load.weight.device.type, "mlu")

        tensor = torch.randn(10).mlu()
        torch.save(tensor, "tensor.pth")
        t_load = torch.load("tensor.pth", map_location="mlu")
        self.assertEqual(t_load.device.type, "mlu")

        dict_a = {"module": m, "tensor": tensor}
        torch.save(dict_a, "dict_a.pth")
        dict_load = torch.load("dict_a.pth", map_location="mlu")
        self.assertEqual(dict_load["module"].weight.device.type, "mlu")
        self.assertEqual(dict_load["tensor"].device.type, "mlu")

    # @unittest.skip("not test")
    def test_distributed(self):
        dist_url = "tcp://127.0.0.1:65501"
        world_size = 1
        rank = 0
        dist.init_process_group(
            backend="cncl", init_method=dist_url, world_size=world_size, rank=rank
        )
        self.assertEqual(torch.distributed.is_available(), True)
        self.assertEqual(torch.distributed.is_initialized(), True)

    # @unittest.skip("not test")
    def test_device_of(self):
        x = torch.randn(5, 5).mlu(1)
        self.assertEqual(torch.mlu.current_device(), 0)
        with torch.mlu.device_of(x):
            self.assertEqual(torch.mlu.current_device(), 1)
        self.assertEqual(torch.mlu.current_device(), 0)

        with torch.mlu.device_of(x):
            self.assertEqual(x.mlu().get_device(), 1)
            torch.mlu.set_device(0)
            self.assertEqual(x.mlu().get_device(), 0)
            with torch.mlu.device_of(x):
                self.assertEqual(x.mlu().get_device(), 1)
            self.assertEqual(x.mlu().get_device(), 0)
            torch.mlu.set_device(1)
        self.assertEqual(x.mlu().get_device(), 0)

    # @unittest.skip("not test")
    def test_grad_scaling_scale(self):
        scaler = torch.mlu.amp.GradScaler(init_scale=2.0)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="mlu:0")
        t1 = torch.full((1,), 4.0, dtype=torch.float32, device="mlu:1")
        # Create some nested iterables of tensors on different devices.
        outputs = (
            t1.clone(),
            (t0.clone(), t1.clone()),
            [t0.clone(), (t1.clone(), t0.clone())],
        )
        outputs = scaler.scale(outputs)
        self.assertTrue(
            outputs[0] == 8.0
            and outputs[1][0] == 8.0
            and outputs[1][1] == 8.0
            and outputs[2][0] == 8.0
            and outputs[2][1][0] == 8.0
            and outputs[2][1][1] == 8.0
        )
        self.assertTrue(scaler._scale.device == t1.device)

    # @unittest.skip("not test")
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
    def test_mlu_get_device_properties(self):
        # Testing the behaviour with None as an argument
        current_device = torch.mlu.current_device()
        current_device_capability = torch.mlu.get_device_properties(current_device)
        device_capability_None = torch.mlu.get_device_properties(None)
        self.assertEqual(
            current_device_capability.total_memory, device_capability_None.total_memory
        )
        self.assertEqual(current_device_capability.major, device_capability_None.major)
        self.assertEqual(current_device_capability.minor, device_capability_None.minor)

        # Testing if the behavior is consistent with torch.mlu.mem_get_info
        self.assertEqual(
            torch.mlu.mem_get_info(current_device)[1],
            torch.mlu.get_device_properties(current_device).total_memory,
        )


if __name__ == "__main__":
    unittest.main()
