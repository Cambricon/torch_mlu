from __future__ import print_function

import sys
import os
import copy

import unittest
import logging
from itertools import product
from numpy import inf, nan
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestCumminOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_cummin(self):
        memory_format_list = [torch.contiguous_format, torch.channels_last]
        shape_list = [(1, 2, 3, 4), (10, 10, 10, 10), (3, 6, 8, 9), (12, 0, 8, 3)]
        dim_list = [0, 1, -2, -1, -2]
        loop_var = [memory_format_list, shape_list, dim_list]
        for memory_format, shape, dim in product(*loop_var):  # pylint: disable=C0200
            x_cpu = torch.randn(shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            x_mlu = self.to_mlu(x_cpu)
            y_value_cpu = torch.zeros(x_cpu.shape, dtype=torch.float)
            y_indice_cpu = torch.zeros(x_cpu.shape, dtype=torch.int64)
            y_value_mlu = self.to_mlu(y_value_cpu)
            y_indice_mlu = self.to_mlu(y_indice_cpu)

            # test cummin(tensor, tensor)
            out_cpu = torch.cummin(x_cpu, dim)
            out_mlu = torch.cummin(x_mlu, dim)
            torch.cummin(x_cpu, dim, out=(y_value_cpu, y_indice_cpu))
            torch.cummin(x_mlu, dim, out=(y_value_mlu, y_indice_mlu))

            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(y_value_cpu.stride() == y_value_mlu.stride())
            self.assertTrue(y_indice_cpu.stride() == y_indice_mlu.stride())

            self.assertTrue(x_cpu.storage_offset() == x_mlu.storage_offset())
            self.assertTensorsEqual(
                out_cpu[0], out_mlu[0].cpu(), 0, use_MSE=True
            )  # float type precision : 0
            self.assertTensorsEqual(
                out_cpu[1], out_mlu[1].cpu(), 0, use_MSE=True
            )  # float type precision : 0
            self.assertTensorsEqual(
                y_value_cpu, y_value_mlu.cpu(), 0, use_MSE=True
            )  # float type precision : 0
            self.assertTensorsEqual(
                y_indice_cpu, y_indice_mlu.cpu(), 0, use_MSE=True
            )  # float type precision : 0

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_backward(self):
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
            out_cpu = x_cpu.cummin(dim)
            out_mlu = input_mlu.cummin(dim)

            grad = torch.randn(shape, dtype=torch.float).to(memory_format=memory_format)

            out_cpu[0].backward(grad, retain_graph=True)
            out_mlu[0].backward(grad.mlu(), retain_graph=True)
            self.assertTensorsEqual(x_cpu.grad, x_mlu.grad, 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_not_dense(self):
        shape_list = [
            (12, 10, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (20, 15, 12, 1, 1, 3),
        ]
        dtype_list = [torch.float, torch.int]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        dim_list = [0, 1, -2, -1, -2]
        loop_var = [shape_list, dtype_list, dim_list]
        for shape, dtype, dim in product(*loop_var):
            x = torch.randn(shape, dtype=torch.float32).to(dtype)
            x_mlu = x.to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            out_cpu = x.cummin(dim=dim)
            out_mlu = x_mlu.cummin(dim=dim)
            self.assertTensorsEqual(out_cpu[0], out_mlu[0].cpu(), 0.0, use_MSE=True)
            self.assertTensorsEqual(out_cpu[1], out_mlu[1].cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_support_dtype(self):
        support_dtype = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.long,
            torch.float16,
            torch.bool,
            torch.float,
            torch.double,
        ]
        support_dtype += [torch.bfloat16] if TEST_BFLOAT16 else []
        for dtype in support_dtype:
            input_c = torch.randn(size=(13, 24), dtype=torch.float32).to(dtype)
            out_values_c = torch.zeros(size=(13, 24), dtype=torch.float32).to(dtype)
            out_indices_c = torch.zeros(size=(13, 24), dtype=torch.float32).to(dtype)

            input_m = self.to_mlu(input_c)
            out_values_m = self.to_mlu(out_values_c)
            out_indices_m = self.to_mlu(out_indices_c)
            if str(dtype) != "torch.float16":
                out_values_c, out_indices_c = torch.cummin(input_c, 1)
            out_values_m, out_indices_m = torch.cummin(input_m, 1)
            if str(dtype) != "torch.float16":
                self.assertTensorsEqual(
                    out_values_c, out_values_m.cpu(), 0, use_MSE=True
                )  # float type precision : 0
                self.assertTensorsEqual(
                    out_indices_c, out_indices_m.cpu(), 0, use_MSE=True
                )  # float type precision : 0
            self.assertEqual(input_m.dtype, out_values_m.dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_not_support_dtype(self):
        not_support_dtype = [torch.complex64, torch.complex32]
        for dtype in not_support_dtype:
            input_c = torch.randn(size=(13, 24), dtype=dtype)
            input_m = self.to_mlu(input_c)
            if dtype == "torch.complex32":
                ref_msg = r"Promotion from ComplexHalf and Float is unsupported."
                with self.assertRaisesRegex(RuntimeError, ref_msg):
                    torch.cummin(input_m, 1)
            if dtype == "torch.complex64":
                ref_msg = r"Can't get a promote dtype."
                with self.assertRaisesRegex(RuntimeError, ref_msg):
                    torch.cummin(input_m, 1)

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_zero(self):
        s = torch.randn(size=[], dtype=torch.float)
        out_cpu = torch.cummin(s, 0)
        out_mlu = torch.cummin(s.to("mlu"), 0)
        self.assertTensorsEqual(
            out_cpu[0], out_mlu[0].cpu(), 0, use_MSE=True
        )  # float type precision : 0
        self.assertTensorsEqual(
            out_cpu[1], out_mlu[1].cpu(), 0, use_MSE=True
        )  # float type precision : 0

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_nan(self):
        input_list = [
            torch.tensor([4, inf, 1.5, -inf, 0, nan, 1]),
            torch.tensor([4, inf, 1.5, inf, 0, nan, inf, 10]),
            torch.tensor([4, nan, inf, 1.5, -inf, -1, nan, 1]),
            torch.tensor([nan, nan, nan, nan]),
        ]
        for x in input_list:
            out_cpu = torch.cummin(x, 0)
            out_mlu = torch.cummin(x.to("mlu"), 0)
            self.assertEqual(out_mlu[0].cpu(), out_cpu[0])
            # CNNL Algorithmic Logic Differences and Instruction Limits
            self.assertNotEqual(out_mlu[1].cpu().sum(), out_cpu[1].sum())

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_scalar(self):
        x = torch.tensor(3.0, requires_grad=True)
        out_cpu = torch.cummin(x, 0)
        out_mlu = torch.cummin(x.to("mlu"), 0)
        self.assertTensorsEqual(
            out_cpu[0], out_mlu[0].cpu(), 0, use_MSE=True
        )  # float type precision : 0
        self.assertTensorsEqual(
            out_cpu[1], out_mlu[1].cpu(), 0, use_MSE=True
        )  # float type precision : 0

    # @unittest.skip("not test")
    @testinfo()
    def test_cummin_exception(self):
        # op shouldn't support values, indices with a dtype, device type or layout
        # different from that of input tensor
        t = torch.randn(10)
        t_mlu = t.to("mlu")
        values = torch.empty(0, dtype=torch.int16)
        indices = torch.empty(0, dtype=torch.int64)
        with self.assertRaisesRegex(
            RuntimeError, "expected scalar_type Float but found Short"
        ):
            torch.cummin(t_mlu, 0, out=(values, indices))

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("69GB")
    def test_cummin_large(self):
        shape = (4, 1025, 1024, 1024)
        dim = 0
        x_cpu = torch.randn(shape, dtype=torch.half)
        x_mlu = self.to_mlu(x_cpu)
        out_cpu = torch.cummin(x_cpu.float(), dim)
        out_mlu = torch.cummin(x_mlu, dim)

        self.assertTensorsEqual(
            out_cpu[0].float(), out_mlu[0].cpu().float(), 0, use_MSE=True
        )  # float type precision : 0
        self.assertTensorsEqual(
            out_cpu[1].float(), out_mlu[1].cpu().float(), 0, use_MSE=True
        )  # float type precision : 0


if __name__ == "__main__":
    run_tests()
