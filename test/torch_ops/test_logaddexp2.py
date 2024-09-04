from __future__ import print_function
import sys
import os

import unittest
import copy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_LARGETENSOR,
    read_card_info,
    largeTensorTest,
)  # pylint: disable=C0413

TEST_BFLOAT16 = read_card_info()
import logging  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestLogAddExp2Op(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_logaddexp2(self):
        shape_list = [(1,), (1, 3), (), (2, 0, 3)]
        type_list = [torch.float32, torch.double]
        for type in type_list:
            for input_shape in shape_list:
                for other_shape in shape_list:
                    input = torch.randn(input_shape, dtype=type)
                    input_copy = copy.deepcopy(input)
                    input_mlu = input_copy.to("mlu")
                    other = torch.randn(other_shape, dtype=type)
                    other_copy = copy.deepcopy(other)
                    other_mlu = other_copy.to("mlu")

                    input.requires_grad = True
                    input_mlu.requires_grad = True
                    other.requires_grad = True
                    other_mlu.requires_grad = True

                    out_cpu = torch.logaddexp2(input, other)
                    out_mlu = torch.logaddexp2(input_mlu, other_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

                    # Test backward
                    grad_cpu = torch.ones(out_cpu.shape, dtype=torch.float)
                    grad_mlu = torch.ones(out_mlu.shape, dtype=torch.float).to("mlu")
                    out_cpu.backward(grad_cpu)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        other.grad, other_mlu.grad.cpu(), 3e-3, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_logaddexp2_channels_last(self):
        shape_list = [(2, 3, 4, 5)]
        type_list = [torch.float32]
        for type in type_list:
            for input_shape in shape_list:
                for other_shape in shape_list:
                    input = torch.randn(input_shape, dtype=type)
                    input.to(memory_format=torch.channels_last)
                    input_copy = copy.deepcopy(input)
                    input_mlu = input_copy.to("mlu")
                    other = torch.randn(other_shape, dtype=type)
                    other.to(memory_format=torch.channels_last)
                    other_copy = copy.deepcopy(other)
                    other_mlu = other_copy.to("mlu")
                    out_cpu = torch.logaddexp2(
                        input.permute([0, 2, 3, 1]), other.permute([0, 2, 3, 1])
                    )
                    out_mlu = torch.logaddexp2(
                        input_mlu.permute([0, 2, 3, 1]), other_mlu.permute([0, 2, 3, 1])
                    )
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    def test_logaddexp2_mixed_input(self):
        shape = (1, 3)
        t1 = torch.float
        t2 = torch.half
        input = torch.randn(shape, dtype=t1)
        input_copy = copy.deepcopy(input)
        input_mlu = input_copy.to("mlu")

        other = torch.randn(shape, dtype=t2)
        other_copy = copy.deepcopy(other)
        other_mlu = other_copy.to("mlu")
        other = other_copy.to(torch.float)

        input.requires_grad = True
        input_mlu.requires_grad = True
        other.requires_grad = True
        other_mlu.requires_grad = True

        out_mlu = torch.logaddexp2(input_mlu, other_mlu)
        out_cpu = torch.logaddexp2(input, other)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

        # Test backward
        grad_cpu = torch.ones(out_cpu.shape, dtype=torch.float)
        grad_mlu = torch.ones(out_mlu.shape, dtype=torch.float).to("mlu")
        out_cpu.backward(grad_cpu)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(
            other.grad, other_mlu.grad.cpu().to(torch.float), 3e-3, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_logaddexp2_large(self):
        shape = (65536, 1024)
        t = torch.float
        input = torch.randn(shape, dtype=t)
        input_copy = copy.deepcopy(input)
        input_mlu = input_copy.to("mlu")

        other = torch.randn(shape, dtype=t)
        other_copy = copy.deepcopy(other)
        other_mlu = other_copy.to("mlu")

        input.requires_grad = True
        input_mlu.requires_grad = True
        other.requires_grad = True
        other_mlu.requires_grad = True

        out_mlu = torch.logaddexp2(input_mlu, other_mlu)
        out_cpu = torch.logaddexp2(input, other)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

        # Test backward
        grad_cpu = torch.ones(out_cpu.shape, dtype=torch.float)
        grad_mlu = torch.ones(out_mlu.shape, dtype=torch.float).to("mlu")
        out_cpu.backward(grad_cpu)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(other.grad, other_mlu.grad.cpu(), 3e-3, use_MSE=True)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_logaddexp2_bfloat16(self):
        shape = (1, 3)
        t = torch.bfloat16
        input = torch.randn(shape, dtype=t)
        input_copy = copy.deepcopy(input)
        input_mlu = input_copy.to("mlu")

        other = torch.randn(shape, dtype=t)
        other_copy = copy.deepcopy(other)
        other_mlu = other_copy.to("mlu")

        input.requires_grad = True
        input_mlu.requires_grad = True
        other.requires_grad = True
        other_mlu.requires_grad = True

        out_mlu = torch.logaddexp2(input_mlu, other_mlu)
        out_cpu = torch.logaddexp2(input, other)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

        # Test backward
        grad_cpu = torch.ones(out_cpu.shape, dtype=torch.float)
        grad_mlu = torch.ones(out_mlu.shape, dtype=torch.float).to("mlu")
        out_cpu.backward(grad_cpu)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(other.grad, other_mlu.grad.cpu(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
