from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413 C0411

logging.basicConfig(level=logging.DEBUG)


class TestMaxUnPoolingOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_max_unpool2d_forward(self):
        shapes = [(14, 64, 64, 128), (1, 3, 32, 32), (1, 1, 256, 256)]
        kernel_v = [2]
        stride_v = [2]
        padding_v = [0, 1]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., :32]]
        dtypes = [torch.float, torch.half]
        loop_var = [shapes, kernel_v, stride_v, padding_v, func_list, dtypes]
        for shape, k, s, p, func, dtype in product(*loop_var):
            input = torch.randn(shape, dtype=dtype)
            input_cpu = input.to(torch.float) if dtype == torch.half else input
            input_mlu = input.mlu()
            pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=p, return_indices=True)
            unpool = nn.MaxUnpool2d(kernel_size=k, stride=s, padding=p)
            output_pool_cpu, indices_cpu = pool(func(input_cpu))
            output_cpu = unpool(func(output_pool_cpu), func(indices_cpu))
            output_pool_mlu, indices_mlu = pool(func(input_mlu))
            indices_mlu_converted = (
                func(indices_mlu).to(torch.int32)
                if dtype == torch.float
                else func(indices_mlu).to(torch.int16)
            )
            output_mlu = unpool(func(output_pool_mlu), indices_mlu_converted)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)

            # torch.nn.functional.max_unpool2d
            output_cpu_F = F.max_unpool2d(
                func(output_pool_cpu),
                func(indices_cpu),
                kernel_size=k,
                stride=s,
                padding=p,
            )
            output_mlu_F = F.max_unpool2d(
                func(output_pool_mlu),
                indices_mlu_converted,
                kernel_size=k,
                stride=s,
                padding=p,
            )
            self.assertTensorsEqual(
                output_cpu_F, output_mlu_F.cpu(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_max_unpool2d_backward(self):
        shapes = [(14, 64, 64, 128), (1, 3, 32, 32), (1, 1, 256, 256)]
        kernel_v = [2]
        stride_v = [2]
        padding_v = [1]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., :32]]
        dtypes = [torch.float, torch.half]
        loop_var = [shapes, kernel_v, stride_v, padding_v, func_list, dtypes]
        for shape, k, s, p, func, dtype in product(*loop_var):
            input = torch.randn(shape, dtype=dtype)
            input_cpu = input.to(torch.float) if dtype == torch.half else input
            input_mlu = copy.deepcopy(input).mlu()
            pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=p, return_indices=True)
            unpool = nn.MaxUnpool2d(kernel_size=k, stride=2, padding=p)

            input_cpu.requires_grad = True
            output_pool_cpu, indices_cpu = pool(func(input_cpu))
            output_cpu = unpool(func(output_pool_cpu), func(indices_cpu))

            grad = torch.randn(shape, dtype=dtype)
            grad_mlu = grad.to("mlu")

            output_cpu.backward(func(grad), retain_graph=True)
            # print("cpu grad is: ", func(input_cpu.grad))

            input_mlu.requires_grad = True
            output_pool_mlu, indices_mlu = pool(func(input_mlu))

            indices_dtype = torch.int32 if dtype == torch.float else torch.int16

            output_mlu = unpool(
                func(output_pool_mlu), func(indices_mlu).to(indices_dtype)
            )

            output_mlu.backward(func(grad_mlu), retain_graph=True)

            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.cpu(), 3e-3, use_MSE=True
            )

            # test torch.nn.functional.max_unpool2d
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            out_cpu_F = F.max_unpool2d(
                func(output_pool_cpu), func(indices_cpu), k, stride=s, padding=p
            )
            out_mlu_F = F.max_unpool2d(
                func(output_pool_mlu),
                func(indices_mlu).to(indices_dtype),
                k,
                stride=s,
                padding=p,
            )
            self.assertTensorsEqual(out_cpu_F, out_mlu_F.cpu(), 3e-3, use_MSE=True)
            grad_F = torch.randn(shape, dtype=dtype)
            output_cpu.backward(func(grad_F.float()), retain_graph=True)
            output_mlu.backward(func(grad_F.mlu()), retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.cpu(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_max_unpool2d_exception(self):
        output, indices = F.max_pool2d(
            torch.randn([1, 1, 4, 4]).mlu(), 2, stride=2, return_indices=True
        )
        ref_msg = "Shape of indices should match shape of input"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            F.max_unpool2d(output, indices.unsqueeze(0), 2)
        ref_msg = "elements in indices should be int16/int32/int64 but got: Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            F.max_unpool2d(output, indices.float(), 2)
        ref_msg = "element in self should be floating types"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            F.max_unpool2d(output.int(), indices.int(), 2)


if __name__ == "__main__":
    unittest.main()
