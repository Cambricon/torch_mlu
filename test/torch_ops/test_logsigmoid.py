from __future__ import print_function

import sys
import os
import logging
import copy
import unittest
from itertools import product
import random
import torch
from torch import nn
from torch import nan, inf
import torch.nn.functional as F
import numpy as np
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

import torch_mlu
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
# TODO(hyl):_is_using_floating_device() unsupport current
# DEVICE_TYPE = torch_mlu._MLUC._is_using_floating_device()
DEVICE_TYPE = True

logging.basicConfig(level=logging.DEBUG)

shape_list = [
    (150),
    (10, 90),
    (45, 50),
    (10, 20, 32),
    (15, 224, 224),
    (2, 32, 128, 128),
    (1, 3, 224, 224),
]
type_list = [(torch.float, 3e-3), (torch.half, 3e-2)]


class TestlogsigmoidOp(TestCase):
    # TODO(huangqipeng): logsigmoid op need bug_fix on MLU290 with CNNLCORE-8953,
    # not testing on this card for now.
    # @unittest.skip("not test")
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    @testinfo()
    def test_logsigmoid(self):
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, shape, func in product(type_list, shape_list, func_list):
            data_type, err = dtype_err
            input = torch.randn(shape, dtype=data_type)
            input_mlu = self.to_mlu(input)
            input_cpu = func(input.float())
            input_mlu = func(input_mlu)

            logsigmoid_layer = nn.LogSigmoid()
            output_cpu = logsigmoid_layer(input_cpu)
            output_mlu = logsigmoid_layer(input_mlu)
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), err, use_MSE=True
            )
            # test function:
            output_cpu_f = F.logsigmoid(input_cpu)
            output_mlu_f = F.logsigmoid(input_mlu)
            self.assertTensorsEqual(
                output_cpu_f, output_mlu_f.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    @testinfo()
    def test_logsigmoid_permute(self):
        for in_shape in [
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16, 4),
            (1, 3, 16, 16, 3, 6),
            (1, 3, 16, 16, 4, 15, 8),
        ]:
            input = torch.randn(in_shape, dtype=torch.float)
            size = np.arange(len(in_shape))
            random.shuffle(size)
            input_mlu = input.to("mlu")
            input = torch.permute(input, tuple(size))
            input_mlu = torch.permute(input_mlu, tuple(size))
            output_cpu = F.logsigmoid(input)
            output_mlu = F.logsigmoid(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_logsigmoid_backward(self):
        for shape in shape_list:
            input = torch.randn(shape, dtype=torch.float, requires_grad=True)
            input_mlu = input.mlu()
            logsigmoid_layer = nn.LogSigmoid()
            out_cpu = logsigmoid_layer(input)
            out_mlu = logsigmoid_layer(input_mlu)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu = grad.mlu()

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(input.grad)
            input.grad.zero_()
            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(input.grad)

            self.assertTensorsEqual(
                out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_logsigmoid_backwark_permute(self):
        input_cpu = torch.randn((4, 3, 2, 1), dtype=torch.float, requires_grad=True)
        out_cpu = F.logsigmoid(input_cpu)
        out_mlu = F.logsigmoid(input_cpu.to("mlu"))
        grad = torch.randn((3, 2, 1, 4), dtype=torch.float)  # test backward
        out_cpu.backward(torch.permute(grad, (3, 0, 1, 2)))
        grad_cpu = copy.deepcopy(input_cpu.grad)
        input_cpu.grad.zero_()
        grad_mlu = grad.mlu().permute(3, 0, 1, 2)
        out_mlu.backward(grad_mlu)
        grad_mlu = input_cpu.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_logsigmoid_boundary_value(self):
        for number in [0, 0.05, 0.0001, -0.0001, -0.01, 999999999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.logsigmoid(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.logsigmoid(self.to_mlu(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_cpu, x, 0)

    @testinfo()
    def test_logsigmoid_special_value(self):
        x = torch.tensor([inf, inf, inf, -inf, -inf, -inf, nan, nan, nan])
        output_cpu = F.logsigmoid(x)
        input_cpu = copy.deepcopy(x)
        output_mlu = F.logsigmoid(self.to_mlu(x))
        self.assertEqual(output_cpu, output_mlu.cpu())
        self.assertEqual(input_cpu, x)

    # @unittest.skip("not test")
    @testinfo()
    def test_logsigmoid_special_case(self):
        shape_list = [(1), (0, 6), ()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            output_cpu = F.logsigmoid(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.logsigmoid(self.to_mlu(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_cpu, x, 0)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_logsigmoid_bfloat16(self):
        shape = [2, 3, 4, 5]
        input = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
        input_mlu = input.mlu()
        logsigmoid_layer = nn.LogSigmoid()
        out_cpu = logsigmoid_layer(input)
        out_mlu = logsigmoid_layer(input_mlu)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        grad_mlu = grad.mlu()

        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(input.grad)
        input.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = copy.deepcopy(input.grad)

        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
