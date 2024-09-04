from __future__ import print_function

import sys
import logging
import os
import copy
import unittest
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestAbsOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_abs_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                out_cpu = torch.abs(x)
                out_mlu = torch.abs(x.to("mlu"))
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                x = self.convert_to_channel_last(x)
                out_cpu = torch.abs(x)
                out_mlu = torch.abs(x.to("mlu"))
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 8),
            (10, 3, 32, 64),
            (2, 3, 8),
            (254, 254, 112, 1, 1, 6),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                x_mlu = x.to("mlu")
                if len(shape) == 4:
                    x = x[:, :, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
                elif len(shape) == 3:
                    x = x[:, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
                out_cpu = torch.abs(x)
                out_mlu = torch.abs(x_mlu)
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                y = (
                    torch.randn(shape, dtype=torch.float32).to(dtype)
                    if dtype != torch.complex64
                    else torch.randn(shape, dtype=torch.float32)
                )
                y_mlu = copy.deepcopy(y).to("mlu")
                out_cpu = torch.abs(x, out=y)
                ori_ptr = y_mlu.data_ptr()
                out_mlu = torch.abs(x.to("mlu"), out=y_mlu)
                out_ptr = y_mlu.data_ptr()
                self.assertEqual(ori_ptr, out_ptr)
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                y = (
                    torch.randn(shape, dtype=torch.float32).to(dtype)
                    if dtype != torch.complex64
                    else torch.randn(shape, dtype=torch.float32)
                )
                x = self.convert_to_channel_last(x)
                y_mlu = copy.deepcopy(y).to("mlu")
                out_cpu = torch.abs(x, out=y)
                ori_ptr = y_mlu.data_ptr()
                out_mlu = torch.abs(x.to("mlu"), out=y_mlu)
                out_ptr = y_mlu.data_ptr()
                self.assertEqual(ori_ptr, out_ptr)
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                x_mlu = x.to("mlu")
                if len(shape) == 4:
                    x = x[:, :, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
                elif len(shape) == 3:
                    x = x[:, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
                y = (
                    torch.randn(shape, dtype=torch.float32).to(dtype)
                    if dtype != torch.complex64
                    else torch.randn(shape, dtype=torch.float32)
                )
                x = self.convert_to_channel_last(x)
                y_mlu = copy.deepcopy(y).to("mlu")
                out_cpu = torch.abs(x, out=y)
                ori_ptr = y_mlu.data_ptr()
                out_mlu = torch.abs(x.to("mlu"), out=y_mlu)
                out_ptr = y_mlu.data_ptr()
                self.assertEqual(ori_ptr, out_ptr)
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                # test out is sliced tensor
                out_cpu = torch.randn([4, 4], dtype=torch.float)
                out_mlu = out_cpu.to("mlu")
                input_cpu = torch.randn([2, 2], dtype=torch.float)
                input_mlu = input_cpu.to("mlu")
                torch.abs(input_cpu, out=out_cpu[:2, :2])
                torch.abs(input_mlu, out=out_mlu[:2, :2])
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_absout_shape_contiguous(self):
        dtype_list = [torch.float, torch.complex64]
        for dtype in dtype_list:
            x = torch.randn(10000, dtype=dtype)
            y = torch.randn(1000, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to("mlu")
            out_cpu = torch.abs(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.abs(x.to("mlu"), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            assert ori_ptr != out_ptr
            if dtype == torch.complex64:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, 0, use_MSE=True)

            x = torch.randn(1000, dtype=dtype)
            y = torch.randn(10000, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to("mlu")
            out_cpu = torch.abs(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.abs(x.to("mlu"), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            if dtype == torch.complex64:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_t_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                out_cpu = x.abs()
                out_mlu = x.to("mlu").abs()
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_t_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int32, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                x = self.convert_to_channel_last(x)
                out_cpu = x.abs()
                out_mlu = x.to("mlu").abs()
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_t_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        dtype_list = [torch.float, torch.complex64, torch.int, torch.long]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                x_mlu = x.to("mlu")
                if len(shape) == 4:
                    x = x[:, :, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
                elif len(shape) == 3:
                    x = x[:, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
                out_cpu = x.abs()
                out_mlu = x_mlu.abs()
                if dtype == torch.complex64:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                else:
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_inplace_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        dtype_list = [torch.float, torch.int]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                x_mlu = copy.deepcopy(x).to("mlu")
                input_ptr = x_mlu.data_ptr()
                x.abs_()
                x_mlu.abs_()
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)
                self.assertEqual(input_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_inplace_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        dtype_list = [torch.float, torch.int]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                x = self.convert_to_channel_last(x)
                x_mlu = copy.deepcopy(x).to("mlu")
                input_ptr = x_mlu.data_ptr()
                x.abs_()
                x_mlu.abs_()
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)
                self.assertEqual(input_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_inplace_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            input_ptr = x_mlu.data_ptr()
            x.abs_()
            x_mlu.abs_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)
            self.assertEqual(input_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_complex(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [torch.complex64, torch.complex128]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=dtype)
                out_cpu = torch.abs(x)
                out_mlu = torch.abs(x.to("mlu"))
                out_cpu_t = x.abs()
                out_mlu_t = x.to("mlu").abs()
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                self.assertTensorsEqual(out_cpu_t, out_mlu_t.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            out_mlu = copy.deepcopy(out).to("mlu")
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.abs(x, out=out)
            torch.abs(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            x_mlu = x.to(dtype).to("mlu")
            x = x.float()
            x.abs_()
            x_mlu.abs_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_int_dtype(self):
        dtype_list = [torch.int, torch.long]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            x_mlu = x.to(dtype).to("mlu")
            x = x.int()
            x.abs_()
            x_mlu.abs_()
            self.assertTensorsEqual(x, x_mlu.cpu().int(), 0.0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB", device="mlu")
    def test_abs_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        dtype_list = [
            torch.float,
        ]
        for shape in shape_list:
            for dtype in dtype_list:
                x = torch.randn(shape, dtype=torch.float32).to(dtype)
                out_cpu = torch.abs(x)
                out_mlu = torch.abs_(x.to("mlu"))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_backward(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)

            out_cpu = torch.abs(x)
            out_mlu = torch.abs(x.to("mlu"))

            grad_in = torch.randn(out_cpu.shape, dtype=out_cpu.dtype)

            out_cpu.backward(grad_in)
            grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()

            out_mlu.backward(grad_in.to("mlu"))
            grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_PYTORCH_11152(self):
        x = torch.randn((1, 4, 1, 64, 64), dtype=torch.float)
        x.as_strided_(x.size(), stride=(4, 1, 4, 256, 4)).requires_grad_()

        out_cpu = torch.abs(x)
        out_mlu = torch.abs(x.to("mlu"))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        grad_in = torch.randn(out_cpu.shape, dtype=out_cpu.dtype)

        out_cpu.backward(grad_in)
        grad_cpu = copy.deepcopy(x.grad)

        x.grad.zero_()

        out_mlu.backward(grad_in.to("mlu"))
        grad_mlu = copy.deepcopy(x.grad)

        self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_abs_exception(self):
        x1 = torch.randn((2, 3, 4, 10), dtype=torch.complex64)
        ref_msg = "In-place abs is not supported for complex tensors."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x1.to("mlu").abs_()

        x2 = torch.randn((2, 3, 4, 10), dtype=torch.complex64)
        out_mlu = torch.randn((2, 3, 4, 10)).int().to("mlu")
        ref_msg = "Float can't be cast to the desired output type Int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.abs(x2.to("mlu"), out=out_mlu)

        x3 = torch.randn((2, 3, 4, 10), dtype=torch.complex64)
        out_mlu = torch.randn((2, 3, 4, 10), dtype=torch.half).to("mlu")
        ref_msg = "For complex input, output must be Float or Double"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.abs(x3.to("mlu"), out=out_mlu)


if __name__ == "__main__":
    run_tests()
