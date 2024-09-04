# pylint: disable=W0223,W0611,R0201,C0413,C0411,C0301,R0402
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import copy
import itertools
from itertools import product
import torch_mlu

import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
import logging

logging.basicConfig(level=logging.DEBUG)

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()


class TestPreluOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_prelu_base(self):
        input = torch.randn(3, 5)
        weight = torch.randn(5)
        input_mlu = self.to_mlu(input)
        weight_mlu = self.to_mlu(weight)
        out_cpu = torch.prelu(input, weight)
        out_mlu = torch.prelu(input_mlu, weight_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prelu(self):
        shape_list = [
            ((3, 224), 1),
            ((1, 6, 224), 6),
            ((4, 3, 224, 224), 3),
            ((1, 3, 224, 224), 3),
            ((3, 224), (224)),
            ((1, 3, 224), (3)),
            ((4, 3, 224, 224), (1,)),
            ((1, 3, 224, 224), (3,)),
        ]
        type_list = [torch.float, torch.half, torch.double]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type, func1, func2 in product(
            shape_list, type_list, func_list, func_list
        ):
            shape1, shape2 = shape
            input = torch.randn(shape1, dtype=type)
            weight = torch.randn(shape2, dtype=type)
            input_mlu = self.to_mlu(input)
            weight_mlu = self.to_mlu(weight)
            input_cpu = func1(input.float())
            weight_cpu = func2(weight.float())
            input_mlu = func1(input_mlu)
            weight_mlu = func2(weight_mlu)
            out_cpu = torch.prelu(input_cpu, weight_cpu)
            out_mlu = torch.prelu(input_mlu, weight_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prelu_backward(self):
        shape_list = [
            [(5), 1, (5)],
            [(3, 2), (3, 1), (3, 2)],
            [(1, 3, 224), (3, 3, 224), 1],
            [1, (2, 6, 1, 1), (2, 6, 224, 224)],
            [(1, 3, 224, 224), 224, (1, 3, 224, 224)],
            [(1, 2, 3, 1, 224), (2, 2, 3, 1, 1), (1, 2, 3, 224, 1)],
        ]
        type_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape_info, type_info, func1, func2, func3 in product(
            shape_list, type_list, func_list, func_list, func_list
        ):
            type, err = type_info
            shape_input, shape_weight, shape_grad = shape_info
            input = torch.randn(shape_input, dtype=type, requires_grad=True)
            weight = torch.randn(shape_weight, dtype=type, requires_grad=True)
            grad = torch.randn(shape_grad, dtype=type)

            input_mlu = self.to_mlu(input)
            weight_mlu = self.to_mlu(weight)
            grad_mlu = self.to_mlu(grad)
            input_cpu = func1(input.float())
            weight_cpu = func2(weight.float())
            grad_cpu = func3(grad.float())
            input_mlu = func1(input_mlu)
            weight_mlu = func2(weight_mlu)
            grad_mlu = func3(grad_mlu)

            x_grad_cpu, weight_grad_cpu = torch.ops.aten._prelu_kernel_backward(
                grad_cpu, input_cpu, weight_cpu
            )
            x_grad_mlu, weight_grad_mlu = torch.ops.aten._prelu_kernel_backward(
                grad_mlu, input_mlu, weight_mlu
            )

            self.assertTensorsEqual(
                x_grad_cpu.float(), x_grad_mlu.cpu().float(), err, use_MSE=True
            )
            self.assertTensorsEqual(
                weight_grad_cpu.float(),
                weight_grad_mlu.cpu().float(),
                err,
                use_MSE=True,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_prelu_exception(self):
        for shape in [(2, 3), (2, 1)]:
            x = torch.randn((1, 4, 6), dtype=torch.float)
            weight = torch.randn(shape, dtype=torch.float)
            msg = (
                r"Mismatch of parameter numbers and input channel size. "
                r"Found parameter numbers = \d and channel size = \d."
            )
            with self.assertRaisesRegex(RuntimeError, msg):
                torch.prelu(x.to("mlu"), weight.to("mlu"))

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("36GB")
    def test_prelu_large(self):
        shape_list = [((5, 1024, 1024, 1024), (1))]
        type_list = [torch.half]
        for shape, dtype in product(shape_list, type_list):
            shape1, shape2 = shape
            input = torch.randn(shape1, dtype=dtype)
            weight = torch.randn(shape2, dtype=dtype)
            input_mlu = self.to_mlu(input)
            weight_mlu = self.to_mlu(weight)
            out_cpu = torch.prelu(input.float(), weight.float())
            out_mlu = torch.prelu(input_mlu, weight_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_prelu_bfloat16(self):
        shape_list = [
            [(5), 1],
            [(3, 2), 2],
            [(1, 3, 224), 3],
            [(2, 6, 224, 224), 6],
            [(1, 3, 224, 224), 3],
            [(1, 2, 3, 224, 224), 2],
        ]
        type_list = [
            (torch.bfloat16, 3e-2),
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape_info, type_info, func1, func2 in product(
            shape_list, type_list, func_list, func_list
        ):
            type, err = type_info
            shape, weight = shape_info
            m = nn.PReLU(weight)
            input = torch.randn(shape, dtype=type, requires_grad=True)
            out_cpu = m(func1(input.float()))
            grad = torch.randn(out_cpu.shape, dtype=type)
            out_cpu.backward(func2(grad.float()))
            x_grad_cpu = copy.deepcopy(input.grad)
            input.grad.zero_()
            m_mlu = m.mlu().to(type)
            out_mlu = m_mlu(func1(self.to_mlu(input)))
            out_mlu.backward(func2(self.to_mlu(grad)))
            x_grad_mlu = input.grad
            self.assertTensorsEqual(
                x_grad_cpu.float(), x_grad_mlu.cpu().float(), err, use_MSE=True
            )


if __name__ == "__main__":
    unittest.main()
