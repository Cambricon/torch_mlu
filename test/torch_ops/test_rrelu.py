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
from common_utils import testinfo, TestCase, read_card_info, skipBFloat16IfNotSupport
import logging

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


def replace_from_cpu(input, cpu_out, mlu_out):
    input_ = input.contiguous().flatten().cpu()
    cpu_out_ = cpu_out.contiguous().flatten()
    mlu_out_ = mlu_out.contiguous().flatten().cpu().float()
    mask = input_ < 0
    mlu_out_[mask] = cpu_out_[mask]
    return mlu_out_.reshape(mlu_out.shape)


def get_uniform_a(input, output):
    input_ = input.contiguous().flatten()
    output_ = output.contiguous().flatten()
    a = 1.0
    for i in range(input_.numel()):
        if input_[i] < 0:
            a = output_[i] / input_[i]
            break
    return a.item()


class TestRReLUOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_rrelu(self):
        shape_list = [
            (1, 6, 224),
            (1, 12, 24, 224),
            (4, 3, 224, 1024),
            (2, 3, 12, 24, 256),
        ]
        type_list = [torch.float, torch.half]
        lower_upper_list = [[0.1, 0.3], [0.1, 0.10001], [0.2, 0.2]]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type, func, lower_upper in product(
            shape_list, type_list, func_list, lower_upper_list
        ):
            lower, upper = lower_upper
            input = torch.randn(shape, dtype=type)
            input_mlu = self.to_mlu(input)
            input_cpu = func(input.float())
            input_mlu = func(input_mlu)

            # training=True: rrelu_with_noise, training=False: leaky_relu
            # case1 training=True, inplace=False
            out_mlu1 = torch.rrelu(input_mlu, lower, upper, True)
            out_cpu1 = torch.rrelu(input_cpu, lower, upper, True)
            out_mlu1_re = replace_from_cpu(input_mlu, out_cpu1, out_mlu1)
            self.assertTensorsEqual(out_cpu1, out_mlu1_re, 0.003, use_MSE=True)

            # case2 training=False, inplace=False
            out_mlu2 = torch.rrelu(input_mlu, lower, upper, False)
            a2 = get_uniform_a(input_mlu, out_mlu2)
            out_cpu2 = torch.rrelu(input_cpu, a2, a2, False)
            self.assertTensorsEqual(
                out_cpu2, out_mlu2.cpu().float(), 0.003, use_MSE=True
            )

            # case3 training=True, inplace=True
            self_cpu1 = copy.deepcopy(input_cpu)
            self_mlu1 = copy.deepcopy(input_mlu)
            torch.rrelu_(self_mlu1, lower, upper, True)
            torch.rrelu_(self_cpu1, lower, upper, True)
            self_mlu1_re = replace_from_cpu(input_mlu, self_cpu1, self_mlu1)
            self.assertTensorsEqual(self_cpu1, self_mlu1_re, 0.003, use_MSE=True)

            # case4 training=False, inplace=True
            self_cpu2 = copy.deepcopy(input_cpu)
            self_mlu2 = copy.deepcopy(input_mlu)
            torch.rrelu_(self_mlu2, lower, upper, False)
            a4 = get_uniform_a(input_mlu, self_mlu2)
            torch.rrelu_(self_cpu2, a4, a4, False)
            self.assertTensorsEqual(
                self_cpu2, self_mlu2.cpu().float(), 0.003, use_MSE=True
            )

    # TODO(hyl): dependency aten::lt.Tensor_out/aten::div.out
    @unittest.skip("not test")
    @testinfo()
    def test_rrelu_backward(self):
        shape_list = [
            (1, 6, 224),
            (1, 12, 24, 224),
            (4, 3, 224, 1024),
            (2, 3, 12, 24, 256),
        ]
        lower_upper_list = [[0.1, 0.3], [0.1, 0.10001], [0.2, 0.2]]
        for shape, lower_upper in product(shape_list, lower_upper_list):
            lower, upper = lower_upper
            x = torch.randn(shape, requires_grad=True)

            out_mlu = torch.rrelu(x.to("mlu"), lower, upper, True)
            a1 = get_uniform_a(x.to("mlu"), out_mlu)
            out_cpu = torch.rrelu(x, a1, a1, True)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

            # backward
            grad = torch.randn(out_cpu.shape)
            out_cpu.backward(grad)
            grad_cpu = x.grad
            x.grad.zero_()
            out_mlu.backward(grad.to("mlu"))
            grad_mlu = x.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # TODO(hyl): fallback to cpu, cann't capture cnnl_rrelu exception
    @unittest.skip("not test")
    @testinfo()
    def test_rrelu_exception(self):
        x = torch.randn((1, 4, 6)).int()
        m = torch.nn.RReLU(0.1, 0.3)
        msg = r"\"cnnl_rrelu\" not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, msg):
            m(x.to("mlu"))

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_rrelu_bfloat16(self):
        shape = [1, 6, 224]
        lower_upper = [0.1, 0.3]
        lower, upper = lower_upper
        x = torch.randn(shape, requires_grad=True, dtype=torch.bfloat16)
        out_mlu = torch.rrelu(x.to("mlu"), lower, upper, True)
        out_cpu = torch.rrelu(x, lower, upper, True)
        out_mlu_re = replace_from_cpu(x.to("mlu"), out_cpu, out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu_re, 0.003, use_MSE=True)

        # backward
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        out_cpu.backward(grad)
        grad_cpu = x.grad
        x.grad.zero_()
        out_mlu.backward(grad.to("mlu"))
        grad_mlu = x.grad
        self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
