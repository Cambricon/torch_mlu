from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import random
import numpy as np
import torch
import torch_mlu
from itertools import product

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestCumsumOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum(self):
        shape_list = [
            (1, 2, 3, 4),
            (10, 10, 10, 10),
            (100, 200),
            (3, 40, 32),
            (1111),
            (99, 30, 40),
            (34, 56, 78, 90),
        ]
        dim_list = [0, 1, -2, 2, -1, 1, 2]
        dtypes = [torch.double, torch.float, torch.long, torch.int]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for t in dtypes:
                x = torch.rand(shape_list[i], dtype=torch.float).to(t)
                x_cpu = x[..., :2]
                x_mlu = self.to_mlu(x)[..., :2]
                y_shape = list(x_cpu.shape)
                y_shape[-1] = y_shape[-1] * 2
                y = torch.rand(y_shape, dtype=torch.float)
                y_cpu = y[..., ::2]
                y_mlu = self.to_mlu(y)[..., ::2]

                # test cumsum(tensor, tensor)
                out_cpu = torch.cumsum(x, dim_list[i])
                out_cpu_1 = torch.cumsum(x_cpu, dim_list[i], out=y_cpu)
                out_mlu_1 = torch.cumsum(self.to_mlu(x), dim_list[i])
                out_mlu_2 = self.to_mlu(x).cumsum(dim_list[i])
                out_mlu_3 = torch.cumsum(self.to_mlu(x), dim_list[i])
                out_mlu_4 = torch.cumsum(x_mlu, dim_list[i], out=y_mlu)
                out_mlu_5 = x_mlu.cumsum(dim_list[i])

                self.assertTrue(x_cpu.stride() == x_mlu.stride())
                self.assertTrue(y_cpu.stride() == y_mlu.stride())

                self.assertTrue(x_cpu.storage_offset() == x_mlu.storage_offset())
                self.assertTensorsEqual(
                    out_cpu, out_mlu_1.cpu(), 0.003, use_MSE=True
                )  # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu, out_mlu_2.cpu(), 0.003, use_MSE=True
                )  # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu, out_mlu_3.cpu(), 0.003, use_MSE=True
                )  # float type precision : 0.003
                self.assertTensorsEqual(
                    y_cpu, y_mlu.cpu(), 0.003, use_MSE=True
                )  # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu_1, out_mlu_4.cpu(), 0.003, use_MSE=True
                )  # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu_1, out_mlu_5.cpu(), 0.003, use_MSE=True
                )  # float type precision : 0.003

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_inplace(self):
        shape_list = [
            (1, 2, 3, 4),
            (10, 10, 10, 10),
            (100, 200),
            (3, 40, 32),
            (1111),
            (99, 30, 40),
            (34, 56, 78, 90),
        ]
        dim_list = [0, 1, -2, 2, -1, 1, 2]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            x = torch.rand(shape_list[i], dtype=torch.float)
            x_mlu = x.mlu()
            x.cumsum_(dim_list[i])
            mlu_ptr = x_mlu.data_ptr()
            x_mlu.cumsum_(dim_list[i])
            self.assertEqual(mlu_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_inplace_no_contiguous(self):
        shape_list = [
            (1, 2, 3, 4),
            (10, 10, 10, 10),
            (100, 200),
            (3, 40, 32),
            (1111),
            (99, 30, 40),
            (34, 56, 78, 90),
        ]
        dim_list = [0, 1, -2, 2, -1, 1, 2]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            x = torch.rand(shape_list[i], dtype=torch.float)
            x_mlu = x.mlu()[..., ::2]
            x_cpu = x[..., ::2]
            x_cpu.cumsum_(dim_list[i])
            mlu_ptr = x_mlu.data_ptr()
            x_mlu.cumsum_(dim_list[i])
            self.assertEqual(mlu_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_inplace_channels_last(self):
        shape_list = [(10, 10, 10, 10), (34, 56, 78, 90)]
        dim_list = [1, 2]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            x = torch.rand(shape_list[i], dtype=torch.float)
            x_mlu = x.mlu().to(memory_format=torch.channels_last)
            x_cpu = x.to(memory_format=torch.channels_last)
            x_cpu.cumsum_(dim_list[i])
            mlu_ptr = x_mlu.data_ptr()
            x_mlu.cumsum_(dim_list[i])
            self.assertEqual(mlu_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_permute(self):
        shape_list = [
            (1, 2, 3, 4),
            (10, 10, 10, 10),
            (100, 200),
            (3, 40, 32),
            (1111,),
            (99, 30, 40),
            (34, 56, 78, 90),
        ]
        shape_list = [(2, 3), (4, 3)]
        dtypes = [torch.double, torch.float]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for t in dtypes:
                x = torch.randn(shape_list[i], dtype=torch.float).to(t)
                x_mlu = x.to("mlu")
                permute_list = np.arange(len(shape_list[i]))
                random.shuffle(permute_list)
                x_mlu_permute = x_mlu.permute(tuple(permute_list))
                x_cpu_permute = x.permute(tuple(permute_list))
                x_mlu_ptr = x_mlu_permute.data_ptr()
                # test cumsum(tensor, tensor)
                out_cpu = torch.cumsum(x_cpu_permute, 0)
                out_mlu = torch.cumsum(x_mlu_permute, 0)

                self.assertTrue(x_cpu_permute.stride() == x_mlu_permute.stride())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())
                self.assertTrue(
                    x_cpu_permute.storage_offset() == x_mlu_permute.storage_offset()
                )
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                x_cpu_permute.cumsum_(0)
                x_mlu_permute.cumsum_(0)
                self.assertTensorsEqual(
                    x_cpu_permute, x_mlu_permute.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_channels_last(self):
        shape_list = [
            (1, 2, 3, 4),
            (10, 10, 10, 10),
            (100, 200, 5, 7),
            (3, 40, 32, 111),
            (99, 30, 40, 45),
            (34, 56, 78, 90),
        ]
        dim_list = [0, 1, -2, 2, -1, 1, 2]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            x = torch.rand(shape_list[i], dtype=torch.float).to(
                memory_format=torch.channels_last
            )

            # test cumsum(tensor, tensor)
            out_cpu = torch.cumsum(x, dim_list[i])
            out_mlu_1 = torch.cumsum(self.to_mlu(x), dim_list[i])
            out_mlu_2 = self.to_mlu(x).cumsum(dim_list[i])
            self.assertTensorsEqual(
                out_cpu, out_mlu_1.cpu(), 0.003, use_MSE=True
            )  # float type precision : 0.003
            self.assertTensorsEqual(
                out_cpu, out_mlu_2.cpu(), 0.003, use_MSE=True
            )  # float type precision : 0.003

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_exception(self):
        DTYPE_NAME_MAP = {
            torch.uint8: "Byte",
            torch.int8: "Char",
            torch.int16: "Short",
            torch.int32: "Int",
            torch.int64: "Long",
            torch.half: "Half",
            torch.float16: "Half",
            torch.float32: "Float",
            torch.float: "Float",
            torch.float64: "Double",
            torch.double: "Double",
            torch.bool: "Bool",
        }

        input_c = torch.randn(size=(13, 24), dtype=torch.float32)
        out_c = torch.zeros(size=(13, 24), dtype=torch.float32)
        input_m = self.to_mlu(input_c)
        out_m = self.to_mlu(out_c)
        not_support_dtype = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.long,
            torch.half,
            torch.bool,
        ]
        for nsd in not_support_dtype:
            ref_msg = r"Expected out tensor to have"
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                torch.cumsum(input_c, 1, dtype=nsd, out=out_c)
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                torch.cumsum(input_m, 1, dtype=nsd, out=out_m)

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_zero(self):
        s = torch.randn(size=[], dtype=torch.float)
        out_cpu = torch.cumsum(s, 0)
        out_mlu = torch.cumsum(s.to("mlu"), 0)
        self.assertTensorsEqual(
            out_cpu, out_mlu.cpu(), 0.003, use_MSE=True
        )  # float type precision : 0.003

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_dtype(self):
        device = "mlu"
        self_tensor = torch.randn(3, 4, 5).to(torch.int8)
        a = torch.cumsum(self_tensor, 0)
        b = torch.cumsum(self_tensor.to(device), 0)
        self.assertEqual(a.dtype, b.dtype)

        self_tensor = torch.randn(3, 4, 5).to(torch.int16)
        c = torch.cumsum(self_tensor, 0)
        d = torch.cumsum(self_tensor.to(device), 0)
        self.assertEqual(c.dtype, d.dtype)

        self_tensor = torch.randn(3, 4, 5).to(torch.int32)
        x = torch.cumsum(self_tensor, 0, dtype=torch.int32)
        y = torch.cumsum(self_tensor.to(device), 0, dtype=torch.int32)
        self.assertEqual(x.dtype, y.dtype)

        self_tensor = torch.randn(3, 4, 5).to(torch.half)
        x = torch.cumsum(self_tensor, 0, dtype=torch.float)
        y = torch.cumsum(self_tensor.to(device), 0, dtype=torch.float)
        self.assertEqual(x.dtype, y.dtype)

        self_tensor = torch.ones(3, 4, 5, dtype=torch.float)
        out_cpu = torch.randn(1)
        out_device = out_cpu.to(device)
        torch.cumsum(self_tensor, 0, out=out_cpu)
        torch.cumsum(self_tensor.to(device), 0, out=out_device)
        self.assertEqual(out_cpu.dtype, out_device.dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_cumsum_backward(self):
        memory_format_list = [torch.contiguous_format, torch.channels_last]
        shape_list = [(1, 2, 3, 4), (10, 10, 10, 10), (3, 5, 7, 9), (12, 0, 8, 3)]
        dim_list = [0, 1, -2, -1, -2]
        loop_var = [memory_format_list, shape_list, dim_list]
        for memory_format, shape, dim in product(*loop_var):  # pylint: disable=C0200
            x_cpu = torch.randn(shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            x_cpu.requires_grad = True
            x_mlu = copy.deepcopy(x_cpu)
            input_mlu = x_mlu.mlu()
            # test cummin(tensor, tensor)
            out_cpu = x_cpu.cumsum(dim)
            out_mlu = input_mlu.cumsum(dim)

            grad = torch.randn(shape, dtype=torch.float).to(memory_format=memory_format)

            out_cpu.backward(grad, retain_graph=True)
            out_mlu.backward(grad.mlu(), retain_graph=True)
            self.assertTensorsEqual(x_cpu.grad, x_mlu.grad, 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("64GB")
    def test_cumsum_large(self):
        shape = [5, 1024, 1024, 1024]
        x = torch.rand(shape, dtype=torch.float)
        out_cpu = torch.cumsum(x, 2)
        out_mlu = torch.cumsum(self.to_mlu(x), 2)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
