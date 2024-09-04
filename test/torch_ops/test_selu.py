from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import numpy as np

import torch
import torch.nn.functional as F
from itertools import product  # pylint: disable=C0411
import random  # pylint: disable=C0411

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


def filter_invalid_value(input):
    type = input.dtype
    shape = input.size()
    a = torch.full(shape, 0.05, dtype=type)
    out = torch.where(torch.eq(input > -0.1, input < 0.05), a, input)
    return out


class TestSeluOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_selu(self):
        shape_list = [(2, 3), (8, 224, 224), (1, 3, 16, 16), (128, 128, 1, 8, 3)]
        type_list = [torch.float]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type, func in product(shape_list, type_list, func_list):
            x = torch.randn(shape)
            x = filter_invalid_value(x).to(type)
            input_cpu = copy.deepcopy(x)
            input_mlu = self.to_mlu(input_cpu)
            input_cpu = func(input_cpu)
            input_mlu = func(input_mlu)
            output_cpu = F.selu(input_cpu)
            output_mlu = F.selu(input_mlu)

            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
            )

            # test inplace operation
            input_cpu_ = copy.deepcopy(x)
            input_mlu_ = self.to_mlu(input_cpu_)
            input_cpu_ = func(input_cpu_)
            input_mlu_ = func(input_mlu_)
            input_mlu_data_ptr = input_mlu_.data_ptr()
            F.selu(input_cpu_, inplace=True)
            F.selu(input_mlu_, inplace=True)
            self.assertEqual(input_mlu_data_ptr, input_mlu_.data_ptr())
            self.assertTensorsEqual(output_mlu.cpu(), input_mlu_.cpu(), 0)
            self.assertTensorsEqual(output_cpu, input_cpu_, 0)
            self.assertTensorsEqual(input_cpu_, input_mlu_.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_selu_backward(self):
        x = torch.tensor([0.1, 0.05, -0.001], dtype=torch.float, requires_grad=True)
        output_cpu = F.selu(x)
        output_mlu = F.selu(self.to_mlu(x))
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
        # backward
        grad = torch.randn(output_cpu.shape)
        output_cpu.backward(grad)
        grad_cpu = x.grad
        x.grad.zero_()
        output_mlu.backward(grad.to("mlu"))
        grad_mlu = x.grad
        self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_selu_permute(self):
        for in_shape in [
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16, 4),
            (1, 3, 16, 16, 3, 6),
            (1, 3, 16, 16, 4, 15, 8),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            input_ = filter_invalid_value(input_)
            size = np.arange(len(in_shape))
            random.shuffle(size)
            input_mlu = self.to_mlu(input_)
            input_ = input_.permute(tuple(size))
            input_mlu = input_mlu.permute(tuple(size))
            input_inplace_ = copy.deepcopy(input_)
            input_mlu_inplace_ = copy.deepcopy(input_mlu)
            output_cpu = F.selu(input_)
            output_mlu = F.selu(input_mlu)
            F.selu(input_inplace_, inplace=True)  # test inplace operation
            F.selu(input_mlu_inplace_, inplace=True)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_inplace_, input_mlu_inplace_.cpu(), 0.003)

            self.assertTrue(output_cpu.storage_offset() == output_mlu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_selu_dtype(self):
        for in_shape in [
            (1),
            (2, 3),
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            # cnnlActivation only support half/float, and cpu kernel don't support half
            dtypes = [torch.float, torch.half, torch.double]
            for dtype in dtypes:
                input_ = torch.randn(in_shape).to(dtype)
                if dtype == torch.half:
                    input_ = input_.float()
                input_ = filter_invalid_value(input_)
                output_cpu = F.selu(input_)
                input_cpu = copy.deepcopy(input_)
                output_mlu = F.selu(self.to_mlu_dtype(input_, dtype))
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
                self.assertTensorsEqual(input_cpu, input_, 0)

                input_mlu = self.to_mlu(input_)  # test inplace operation
                input_mlu_data_ptr = input_mlu.data_ptr()
                F.selu(input_cpu, inplace=True)
                F.selu(input_mlu, inplace=True)
                self.assertTensorsEqual(output_cpu, input_mlu.cpu().float(), 0.003)
                self.assertTensorsEqual(input_cpu, input_mlu.cpu().float(), 0.003)
                self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_selu_boundary_value(self):
        for number in [0, 0.05, -0.01, 999999999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.selu(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.selu(self.to_mlu(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_cpu, x, 0)

            input_cpu = copy.deepcopy(x)
            input_mlu = self.to_mlu(x)  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.selu(input_cpu, inplace=True)
            F.selu(input_mlu, inplace=True)
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_selu_special_case(self):
        shape_list = [(1), (0, 6), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x = filter_invalid_value(x)
            output_cpu = F.selu(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.selu(self.to_mlu(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_cpu, x, 0)

            input_cpu = copy.deepcopy(x)
            input_mlu = self.to_mlu(x)  # test inplace operation
            input_mlu_data_ptr = input_mlu.data_ptr()
            F.selu(input_cpu, inplace=True)
            F.selu(input_mlu, inplace=True)
            self.assertTensorsEqual(output_cpu, input_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_selu_bfloat16(self):
        x = torch.tensor([0.1, 0.05, -0.001], dtype=torch.bfloat16, requires_grad=True)
        x_mlu = x.to("mlu")
        out_cpu = F.selu(x)
        out_mlu = F.selu(x_mlu)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        grad_mlu = grad.to("mlu")
        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = copy.deepcopy(x.grad)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003)
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.003)


if __name__ == "__main__":
    unittest.main()
