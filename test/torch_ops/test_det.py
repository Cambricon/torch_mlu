from __future__ import print_function
import sys
import os

import unittest
import copy

from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    skipBFloat16IfNotSupport,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_openblas_info,
)

import logging  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)
USE_OPENBLAS = read_openblas_info()


class TestdetOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_det_channels_last(self):
        shape_list = [(5, 3, 5, 5)]
        dtype_list = [torch.float32, torch.double]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype, shape, func in product(dtype_list, shape_list, func_list):
            x = torch.rand(shape, dtype=dtype)
            x_copy = copy.deepcopy(x)
            x_mlu = x_copy.to("mlu")
            x = x.to(memory_format=torch.channels_last)
            x_mlu = x_mlu.to(memory_format=torch.channels_last)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = torch.det(func(x.float()))
            out_mlu = torch.det(func(self.to_mlu_dtype(x, dtype)))
            out_cpu.backward(out_cpu)
            out_mlu.backward(out_mlu)

            for i in range(out_cpu.numel()):
                cpu_res = out_cpu.contiguous().view(-1)
                mlu_res = out_mlu.cpu().contiguous().view(-1)
                if torch.isnan(cpu_res[i]):
                    continue
                self.assertTensorsEqual(
                    cpu_res[i], mlu_res.cpu()[i], 1e-3, use_MSE=True
                )
        # test empty input
        shape_list = [(0, 0), (3, 0, 0), (0, 3, 3), (3, 0, 3, 3), (5, 3, 0, 0)]
        for shape in shape_list:
            x = torch.randn(shape)
            out_cpu = torch.det(x)
            out_mlu = torch.det(x.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_linalg_det_out(self):
        shape_list = [(5, 3, 5, 5)]
        dtype_list = [torch.float32, torch.double]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype, shape, func in product(dtype_list, shape_list, func_list):
            x = torch.rand(shape, dtype=dtype)
            x_mlu = x.to("mlu")
            out = torch.empty((0), dtype=dtype, device="mlu")
            x = x.to(memory_format=torch.channels_last)
            x_mlu = x_mlu.to(memory_format=torch.channels_last)
            res_cpu = torch.linalg.det(func(x.float()))
            res_mlu = torch.linalg.det(func(self.to_mlu_dtype(x, dtype)), out=out)

            for i in range(res_cpu.numel()):
                cpu_res = res_cpu.contiguous().view(-1)
                mlu_res = res_mlu.cpu().contiguous().view(-1)
                out = out.cpu().contiguous().view(-1)
                if torch.isnan(cpu_res[i]):
                    continue
                self.assertTensorsEqual(
                    cpu_res[i], mlu_res.cpu()[i], 1e-3, use_MSE=True
                )
                self.assertTensorsEqual(out[i], mlu_res.cpu()[i], 1e-3, use_MSE=True)

        # test empty input
        shape_list = [(0, 0), (3, 0, 0), (0, 3, 3), (3, 0, 3, 3), (5, 3, 0, 0)]
        for shape in shape_list:
            x = torch.randn(shape)
            out = torch.randn([2, 5, 3]).mlu()
            out_cpu = torch.linalg.det(x)
            out_mlu = torch.linalg.det(x.to("mlu"), out=out)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(out_cpu, out.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_det_backward(self):
        test_inputs = [
            torch.randn((5, 3, 5, 5)).requires_grad_(),
            torch.randn((5, 5)).requires_grad_(),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]).requires_grad_(),
            torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]).requires_grad_(),
        ]
        for x in test_inputs:
            x_mlu = x.detach().clone().mlu().requires_grad_()
            res = torch.det(x)
            res_mlu = torch.det(x_mlu)
            w = torch.randn_like(res)
            w_mlu = w.detach().clone().mlu()
            res.backward(w)
            res_mlu.backward(w_mlu)
            out_cpu = x.grad
            out_mlu = x_mlu.grad
            for i in range(out_cpu.numel()):
                cpu_res = out_cpu.contiguous().view(-1)
                mlu_res = out_mlu.cpu().contiguous().view(-1)
                if torch.isnan(cpu_res[i]):
                    continue
                self.assertTensorsEqual(
                    cpu_res[i], mlu_res.cpu()[i], 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_det_exception(self):
        a = torch.randn(64, 63).to("mlu")
        msg = f"linalg.det: A must be batches of square matrices, but they are 64 by 63 matrices"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.det(a)

        a = torch.randn(64, 64).to("mlu")
        out = torch.randn(64, 64, dtype=torch.half).to("mlu")
        msg = f"Expected out tensor to have dtype float, but got c10::Half instead"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.linalg.det(a, out=out)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipIf(
        USE_OPENBLAS, "torch.det on CPU using openblas might core dumped or timeout"
    )
    @largeTensorTest("21GB")
    def test_det_large(self):
        # Oversized shape may cause time-out
        shape = (8, 1025, 512, 512)
        dtype = torch.float
        x = torch.rand(shape, dtype=dtype)
        out_cpu = torch.det(x.float())
        out_mlu = torch.det(x.mlu())

        for i in range(out_cpu.numel()):
            out_cpu = out_cpu.view(-1)
            out_mlu = out_mlu.cpu().view(-1)
            if torch.isnan(out_cpu[i]):
                continue
            self.assertTensorsEqual(
                out_cpu[i], out_mlu[i].cpu(), 3e-3, use_MSE=True, allow_inf=True
            )


if __name__ == "__main__":
    run_tests()
