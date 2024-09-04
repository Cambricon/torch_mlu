from __future__ import print_function
import logging
import unittest
import sys
import os
import copy
import torch

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


class TestExpOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_exp_contiguous(self):
        shape_list = [(16, 384, 3072), (16, 0, 88)]
        data_types = [torch.float, torch.half, torch.double]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.exp(x)
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_channel_last(self):
        shape_list = [(2, 3, 3, 4)]
        data_types = [torch.float, torch.half, torch.double]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                out_cpu = torch.exp(x)
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_not_dense(self):
        shape_list = [(2, 3, 3, 4)]
        data_types = [torch.float, torch.half, torch.double]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.exp(x[:, :, :, :2])
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type)[:, :, :, :2])
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_inplace_contiguous(self):
        shape_list = [(27), (13, 78), (16, 384, 3072), (13, 24, 35, 46), (16, 0, 88)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.exp_(x)
                torch.exp_(x_mlu)
                self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_inplace_channel_last(self):
        shape_list = [(2, 3, 3, 4)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.exp_(x)
                torch.exp_(x_mlu)
                self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_inplace_not_dense(self):
        shape_list = [(2, 3, 3, 4)]
        data_types = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.exp_(x[:, :, :, :2])
                torch.exp_(x_mlu[:, :, :, :2])
                self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_out(self):
        shape_list = [(27), (13, 78), (16, 384, 3072), (13, 24, 35, 46), (16, 0, 88)]
        data_types = [torch.float, torch.half]
        out_shapes = [(100, 10), (1), (20, 20, 60, 100), (77, 0, 88, 99)]
        for out_shape in out_shapes:
            for shape in shape_list:
                for data_type in data_types:
                    x = torch.randn(shape, dtype=torch.float)
                    x_mlu = self.to_mlu_dtype(x, data_type)
                    out_cpu = torch.randn(out_shape, dtype=torch.float)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                    torch.exp(x, out=out_cpu)
                    torch.exp(x_mlu, out=out_mlu)
                    self.assertTensorsEqual(
                        out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_permute(self):
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
            torch.exp(x, out=out)
            torch.exp(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_type(self):
        shape_list = [(1, 3, 16, 16)]
        type_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.short,
            torch.bool,
        ]
        for shape in shape_list:
            for type in type_list:
                x_cpu = torch.randn(shape).to(type)
                x_mlu = self.to_mlu(x_cpu)
                if type == torch.half:
                    x_cpu = x_cpu.float()
                out_cpu = torch.exp(x_cpu)
                out_mlu = torch.exp(x_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_exp_backward(self):
        shape_list = [
            (66),
            (39, 48),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (11, 13, 16, 18, 20, 23),
        ]
        type_list = [torch.float, torch.half]
        for shape in shape_list:
            for data_type in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(data_type)
                x_mlu = x.to("mlu")

                # use float on cpu kernel
                out_cpu = x_0.exp()
                out_mlu = x_mlu.exp()

                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.to("mlu")

                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)

                self.assertTensorsEqual(
                    out_grad_cpu,
                    out_grad_mlu.cpu().float()
                    if data_type == torch.half
                    else out_grad_mlu.cpu(),
                    0.003,
                    use_MSE=True,
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_exp_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        data_types = [torch.half]
        for shape in shape_list:
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.exp(x)
                out_mlu = torch.exp(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    unittest.main()
