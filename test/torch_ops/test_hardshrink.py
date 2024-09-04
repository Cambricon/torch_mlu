from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import random
import numpy as np

import torch
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    TEST_BFLOAT16,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestHardshrinkOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_hardshrink(self):
        for in_shape in [
            (11, 33),
            (8, 111, 131),
            (1, 1, 1, 1),
            (7, 3, 9, 16),
            (1, 2, 0, 3),
            (1, 90, 91, 8, 96, 2),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            input_cpu = copy.deepcopy(input_)
            lambds = [0.1, 0.01, 0.023, 0.5, 1.5]
            for lambd in lambds:
                output_cpu = F.hardshrink(input_cpu, lambd)
                output_mlu = F.hardshrink(input_.to("mlu"), lambd)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0, use_MSE=True)
                self.assertTensorsEqual(input_cpu, input_, 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardshrink_boundary_value(self):
        for number in [0, 0.0001, 99999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.hardshrink(x)
            output_mlu = F.hardshrink(x.to("mlu"))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardshrink_backward(self):
        for shape in [
            (9, 17),
            (8, 224, 224),
            (1, 1, 1, 1),
            (8, 3, 16, 16),
            (2, 3, 0, 4),
        ]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = F.hardshrink(x)
            out_mlu = F.hardshrink(x.to("mlu"))
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu.backward(grad.to("mlu"))
            grad_mlu = x.grad
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=True)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardshrink_dtype(self):
        for in_shape in [
            (1),
            (2, 3),
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
            (1, 90, 91, 8, 96, 2),
        ]:
            for dtype in [torch.float, torch.half, torch.double]:
                x = torch.randn(in_shape, dtype=dtype, requires_grad=True)
                out_cpu = F.hardshrink(x, lambd=0.01)
                out_mlu = F.hardshrink(x.to("mlu"), lambd=0.01)
                grad = torch.randn(out_cpu.shape, dtype=dtype)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()
                out_mlu.backward(grad.to("mlu"))
                grad_mlu = x.grad
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardshrink_channels_last(self):
        for in_shape in [
            (3, 8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16),
            (1, 3, 16, 16),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float).to(
                memory_format=torch.channels_last
            )
            output_cpu = F.hardshrink(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.hardshrink(input_.to("mlu"))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardshrink_not_dense(self):
        for in_shape in [
            (2, 4),
            (8, 224, 224),
            (1, 1, 1, 8),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 10),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            input_mlu = self.to_mlu(input_)[..., :2]
            input_cpu = input_[..., :2]
            output_cpu = F.hardshrink(input_cpu)
            input_cpu_1 = copy.deepcopy(input_cpu)
            output_mlu = F.hardshrink(input_mlu)
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input_cpu_1, input_cpu, 0)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_hardshrink_bfloat16(self):
        shape = [1, 90, 91, 8, 96, 2]
        x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
        out_cpu = F.hardshrink(x, lambd=0.01)
        out_mlu = F.hardshrink(x.to("mlu"), lambd=0.01)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        out_cpu.backward(grad)
        grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad.to("mlu"))
        grad_mlu = x.grad
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("69GB")
    def test_hardshrink_large(self):
        shape = [4, 1025, 1024, 1024]
        x = torch.randn(shape, requires_grad=True)
        out_cpu = F.hardshrink(x, lambd=0.01)
        out_mlu = F.hardshrink(x.to("mlu"), lambd=0.01)
        grad = torch.randn(out_cpu.shape)
        out_cpu.backward(grad)
        grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad.to("mlu"))
        grad_mlu = x.grad
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
