from __future__ import print_function

import sys
import os
import unittest
import logging
import itertools
from itertools import product
import random
import copy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestUnfoldOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_unfold(self):
        dtype_list = [
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        shape_list = [
            (99,),
            (12, 24),
            (3, 18, 9),
            (15, 25, 35, 2),
            (5, 16, 9, 10, 2),
            (5, 16, 9, 10, 2, 4),
        ]
        for shape, dtype, func in product(shape_list, dtype_list, func_list):
            for dim in range(-len(shape), len(shape)):
                size = random.randint(1, shape[dim])
                step = random.randint(1, size)
                input = torch.randn(shape).to(dtype)
                input_cpu = func(input)
                input_mlu = func(input.to("mlu"))
                output_cpu = input_cpu.unfold(dim, size, step)
                output_mlu = input_mlu.unfold(dim, size, step)
                self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_unfold_backward(self):
        shape_list = [(12, 24), (3, 18, 9), (15, 25, 35, 2), (5, 16, 9, 10, 2)]
        for shape in shape_list:
            for dim in range(-len(shape), len(shape)):
                size = random.randint(1, shape[dim])
                step = random.randint(1, size)
                input_cpu = torch.randn(shape)
                input_mlu = copy.deepcopy(input_cpu).to("mlu")
                input_cpu.requires_grad = True
                input_mlu.requires_grad = True
                output_cpu = input_cpu.unfold(dim, size, step)
                output_mlu = input_mlu.unfold(dim, size, step)
                grad_cpu = torch.randn(output_cpu.shape, dtype=torch.float)
                grad_mlu = copy.deepcopy(grad_cpu).to("mlu")
                output_cpu.backward(grad_cpu)
                output_mlu.backward(grad_mlu)
                self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
                self.assertTensorsEqual(
                    input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0.003
                )

        # test 0-dim
        input_cpu = torch.randn(())
        input_mlu = copy.deepcopy(input_cpu).to("mlu")
        input_cpu.requires_grad = True
        input_mlu.requires_grad = True
        output_cpu = input_cpu.unfold(0, 1, 2)
        output_mlu = input_mlu.unfold(0, 1, 2)
        grad_cpu = torch.randn(output_cpu.shape, dtype=torch.float)
        grad_mlu = copy.deepcopy(grad_cpu).to("mlu")
        output_cpu.backward(grad_cpu)
        output_mlu.backward(grad_mlu)
        self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0.003
        )

        # test numel=0
        input_cpu = torch.randn((6, 0, 2))
        input_mlu = copy.deepcopy(input_cpu).to("mlu")
        input_cpu.requires_grad = True
        input_mlu.requires_grad = True
        output_cpu = input_cpu.unfold(0, 1, 2)
        output_mlu = input_mlu.unfold(0, 1, 2)
        grad_cpu = torch.randn(output_cpu.shape, dtype=torch.float)
        grad_mlu = copy.deepcopy(grad_cpu).to("mlu")
        output_cpu.backward(grad_cpu)
        output_mlu.backward(grad_mlu)
        self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0.003
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_unfold_backward_combination(self):
        shape = (4, 3, 16, 32)
        dim_list = list(range(-len(shape), len(shape)))
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        param_list = [dim_list, func_list]
        for dim, func in itertools.product(*param_list):
            x = torch.randn(shape, dtype=torch.float)
            x.requires_grad = True
            size = random.randint(1, func(x).shape[dim])
            step = random.randint(1, size)
            out_cpu = func(x).unfold(dim, size, step)
            grad = torch.randn_like(out_cpu)
            out_cpu.backward(grad)
            x_grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()

            out_mlu = func(self.to_mlu(x)).unfold(dim, size, step)
            out_mlu.backward(self.to_mlu(grad))
            x_grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTensorsEqual(
                x_grad_cpu.float(), x_grad_mlu.float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_unflod_bfloat16(self):
        shape = [2, 3, 4, 5]
        size = random.randint(1, 4)
        step = random.randint(1, size)
        input_cpu = torch.randn(shape).to(torch.bfloat16)
        input_mlu = input_cpu.to("mlu")
        input_cpu.requires_grad = True
        input_mlu.requires_grad = True
        output_cpu = input_cpu.unfold(2, size, step)
        output_mlu = input_mlu.unfold(2, size, step)
        grad_cpu = torch.randn(output_cpu.shape, dtype=torch.bfloat16)
        grad_mlu = copy.deepcopy(grad_cpu).to("mlu")
        output_cpu.backward(grad_cpu)
        output_mlu.backward(grad_mlu)
        self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0)
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0.003
        )


if __name__ == "__main__":
    unittest.main()
