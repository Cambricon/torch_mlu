from __future__ import print_function

import sys
import os
import unittest
import logging
import copy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt(self):
        shape_list = [(12, 1024, 2, 2, 4), (2, 3, 4), (20, 254, 112, 1, 1, 3), ()]
        data_types = [torch.float, torch.half, torch.double]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                out_cpu = torch.rsqrt(x)
                out_mlu = torch.rsqrt(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_channel_last(self):
        shape_list = [(12, 1024, 2, 2), (2, 3, 4, 5)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape).to(memory_format=torch.channels_last) + 0.01
                out_cpu = torch.rsqrt(x)
                out_mlu = torch.rsqrt(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_not_dense(self):
        shape_list = [(12, 1024, 2, 2, 4), (2, 3, 4), (20, 254, 112, 1, 1, 3)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape) + 0.01
                out_cpu = torch.rsqrt(x[:, ..., :2])
                out_mlu = torch.rsqrt(self.to_mlu_dtype(x, data_type)[:, ..., :2])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_inplace(self):
        shape_list = [(), (2, 3, 4), (64, 3, 224), (12, 4, 6, 7, 2), (2, 3, 6, 2, 1, 2)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                x_mlu = self.to_mlu_dtype(x, data_type)
                x_ptr = x_mlu.data_ptr()
                torch.rsqrt_(x)
                torch.rsqrt_(x_mlu)
                self.assertEqual(x_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_inplace_channel_last(self):
        shape_list = [(64, 3, 224, 3), (12, 4, 6, 7)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape).to(memory_format=torch.channels_last) + 0.01
                x_mlu = self.to_mlu_dtype(x, data_type)
                x_ptr = x_mlu.data_ptr()
                torch.rsqrt_(x)
                torch.rsqrt_(x_mlu)
                self.assertEqual(x_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_inplace_not_dense(self):
        shape_list = [
            (2, 3, 4),
            (64, 3, 224),
            (
                12,
                4,
                6,
                7,
            ),
            (2, 3, 6, 2, 1, 2, 4),
        ]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                x_mlu = self.to_mlu_dtype(x, data_type)
                x_ptr = x_mlu.data_ptr()
                torch.rsqrt_(x[:, ..., :2])
                torch.rsqrt_(x_mlu[:, ..., :2])
                self.assertEqual(x_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_out(self):
        shape_list = [(12, 1024, 2, 2, 4), (2, 3, 4), (20, 254, 112, 1, 1, 3), ()]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                out_cpu = torch.zeros((1)).float()
                out_mlu = torch.zeros((1)).to(data_type).to("mlu")
                torch.rsqrt(x, out=out_cpu)
                torch.rsqrt(self.to_mlu_dtype(x, data_type), out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                out_cpu = torch.zeros(shape).float()
                out_mlu = torch.zeros(shape).to(data_type).to("mlu")
                torch.rsqrt(x, out=out_cpu)
                torch.rsqrt(self.to_mlu_dtype(x, data_type), out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.rand(shape_list[i], dtype=torch.float) + 0.01
            out = torch.rand(shape_list[i], dtype=torch.float) + 0.01
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.rsqrt(x, out=out)
            torch.rsqrt(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_floating_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.rand((2, 3, 4, 5, 6), dtype=torch.float) + 0.00005
            x_mlu = self.to_mlu_dtype(x, dtype)
            out_cpu = torch.rsqrt(x)
            out_mlu = torch.rsqrt(x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_integral_dtype(self):
        dtype_list = [torch.uint8, torch.int8, torch.short, torch.int, torch.long]
        for dtype in dtype_list:
            x = torch.testing.make_tensor((2, 3, 4, 5, 6), dtype=dtype, device="cpu")
            x_mlu = x.to("mlu")
            output_cpu = torch.rsqrt(x)
            output_mlu = torch.rsqrt(x_mlu)
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu(), 3e-3, allow_inf=True, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsqrt_backward(self):
        shape_list = [
            (66),
            (39, 48),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (11, 13, 16, 18, 20, 23),
        ]
        type_list = [torch.float]
        for shape in shape_list:
            for data_type in type_list:
                x_0 = torch.randn(shape, dtype=data_type)
                x_mlu = x_0.to("mlu")
                x_0.requires_grad_(True)
                x_mlu.requires_grad_(True)
                out_cpu = torch.rsqrt(x_0)
                out_mlu = torch.rsqrt(x_mlu)
                out_cpu.backward(torch.ones_like(out_cpu))
                out_mlu.backward(torch.ones_like(out_mlu))
                self.assertTensorsEqual(
                    x_0.grad, x_mlu.grad.cpu(), 0.003, allow_inf=True, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_rsqrt_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        data_types = [torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.rand(shape, dtype=torch.float) + 0.01
                out_cpu = torch.rsqrt(x)
                out_mlu = torch.rsqrt(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
