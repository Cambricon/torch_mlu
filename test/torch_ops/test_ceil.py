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
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestCeilOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
            (0, 6),
        ]
        dtypes = [torch.double, torch.float]
        for shape in shape_list:
            for t in dtypes:
                x = torch.randn(shape, dtype=torch.float).to(t)
                out_cpu = torch.ceil(x)
                out_mlu = torch.ceil(x.to("mlu"))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_backward(self):
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            x_mlu = self.to_device(x)

            out_cpu = torch.ceil(x)
            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_cpu.backward(grad)
            x_grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()

            out_mlu = torch.ceil(x_mlu)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTensorsEqual(
                x_grad_cpu.float(), x.grad.float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        for shape in shape_list:
            for t in [torch.double, torch.float]:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x = self.convert_to_channel_last(x)
                out_cpu = torch.ceil(x)
                out_mlu = torch.ceil(x.to("mlu"))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 8),
            (10, 3, 32, 64),
            (2, 3, 8),
            (254, 254, 112, 1, 1, 6),
        ]
        for shape in shape_list:
            for t in [torch.double, torch.float]:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x_mlu = x.to("mlu")
                if len(shape) == 4:
                    x = x[:, :, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
                elif len(shape) == 3:
                    x = x[:, :, : int(shape[-1] / 2)]
                    x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
                out_cpu = torch.ceil(x)
                out_mlu = torch.ceil(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceilout_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to("mlu")
            out_cpu = torch.ceil(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.ceil(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceilout_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            y_mlu = copy.deepcopy(y).to("mlu")
            out_cpu = torch.ceil(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.ceil(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceilout_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            y = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            y_mlu = copy.deepcopy(y).to("mlu")
            out_cpu = torch.ceil(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.ceil(x_mlu, out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceilout_shape_contiguous(self):
        x = torch.randn(10000, dtype=torch.float)
        y = torch.randn(1000, dtype=torch.float)
        y_mlu = copy.deepcopy(y).to("mlu")
        out_cpu = torch.ceil(x, out=y)
        ori_ptr = y_mlu.data_ptr()
        out_mlu = torch.ceil(self.to_mlu(x), out=y_mlu)
        out_ptr = y_mlu.data_ptr()
        assert ori_ptr != out_ptr
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        x = torch.randn(1000, dtype=torch.float)
        y = torch.randn(10000, dtype=torch.float)
        y_mlu = copy.deepcopy(y).to("mlu")
        out_cpu = torch.ceil(x, out=y)
        ori_ptr = y_mlu.data_ptr()
        out_mlu = torch.ceil(self.to_mlu(x), out=y_mlu)
        out_ptr = y_mlu.data_ptr()
        self.assertEqual(ori_ptr, out_ptr)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_t_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = x.ceil()
            out_mlu = self.to_mlu(x).ceil()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_t_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            out_cpu = x.ceil()
            out_mlu = self.to_mlu(x).ceil()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_t_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            out_cpu = x.ceil()
            out_mlu = x_mlu.ceil()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_inplace_contiguous(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            x.ceil_()
            x_mlu.ceil_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_inplace_channel_last(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x = self.convert_to_channel_last(x)
            x_mlu = copy.deepcopy(x).to("mlu")
            x.ceil_()
            x_mlu.ceil_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_inplace_not_dense(self):
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
            x.ceil_()
            x_mlu.ceil_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_permute(self):
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
            torch.ceil(x, out=out)
            torch.ceil(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ceil_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x = x.float()
            x.ceil_()
            x_mlu.ceil_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_ceil_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        dtypes = [torch.float]
        for shape in shape_list:
            for t in dtypes:
                x = torch.randn(shape, dtype=t)
                # input must be [-2^23 + 1, 2^23 - 1] because of cnnlCeil's limit
                x_cpu = x.clamp(-2 ^ 23 + 1, 2 ^ 23 - 1)
                out_cpu = torch.ceil(x_cpu)
                out_mlu = torch.ceil(x_cpu.to("mlu"))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_ceil_bfloat16(self):
        x = torch.randn((2, 3, 4, 5, 6), dtype=torch.bfloat16)
        x_mlu = self.to_mlu_dtype(x, torch.bfloat16)
        x.ceil_()
        x_mlu.ceil_()
        self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
