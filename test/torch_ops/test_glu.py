from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
import torch.nn as NN

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestGluOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_glu(self):
        shape_list = [
            (50),
            (35, 46),
            (16, 28, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 14, 22),
            (6, 7, 8, 9, 10, 12),
            (16, 17, 18, 19, 20, 22),
        ]
        type_list = [torch.float, torch.half]
        mode_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        dim_list = [-1, -2]
        list_list = [type_list, shape_list, mode_list, dim_list]

        for dtype, shape, mode, dim in product(*list_list):
            if isinstance(shape, list) and len(shape) > 2:
                m = NN.GLU(dim)
            else:
                m = NN.GLU(-1)
            x_0 = torch.randn(shape, dtype=torch.float, requires_grad=False)
            x = x_0.to(dtype)
            x_mlu = x.to("mlu")
            x_0 = mode(x_0)
            x_0.requires_grad = True
            x_mlu = mode(x_mlu)
            x_mlu.requires_grad = True

            out_cpu = m(x_0)
            out_mlu = m(x_mlu)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.to("mlu")

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x_0.grad)
            x_0.grad.zero_()
            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x_mlu.grad)

            self.assertTensorsEqual(
                out_cpu,
                out_mlu.cpu().float(),
                0.03 if dtype == torch.half else 0.003,
                use_MSE=True,
            )

            self.assertTensorsEqual(
                out_grad_cpu,
                out_grad_mlu.cpu().float(),
                0.03 if dtype == torch.half else 0.003,
                use_MSE=True,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_glu_backward_stride_channel_last(self):
        input_shapes = [(1, 128, 1, 128)]
        grad_shapes = [(1, 128, 1, 64)]
        strides = [(128, 1, 8912, 128)]
        for input_shape, grad_shape, stride in zip(input_shapes, grad_shapes, strides):
            input_t = torch.randn(input_shape)
            input_cpu = torch.nn.Parameter(
                input_t.to(memory_format=torch.channels_last)
            )
            input_mlu = torch.nn.Parameter(
                input_t.to("mlu").to(memory_format=torch.channels_last)
            )
            cpu_out = torch.nn.functional.glu(input_cpu)
            mlu_out = torch.nn.functional.glu(input_mlu)

            grad = torch.randn(grad_shape)
            cpu_out.backward(grad.as_strided(grad_shape, stride))
            mlu_out.backward(grad.to("mlu").as_strided(grad_shape, stride))

            self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_cpu.grad, input_mlu.grad.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_glu_bfloat16(self):
        input_shapes = [(1, 128, 1, 128)]
        grad_shapes = [(1, 128, 1, 64)]
        strides = [(128, 1, 8912, 128)]
        for input_shape, grad_shape, stride in zip(input_shapes, grad_shapes, strides):
            input_t = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
            input_t_mlu = input_t.mlu()
            grad = torch.randn(grad_shape, dtype=torch.bfloat16)
            grad_mlu = grad.mlu()

            cpu_out = torch.nn.functional.glu(input_t.float())
            cpu_out.backward(grad.float().as_strided(grad_shape, stride))

            input_grad_cpu = copy.deepcopy(input_t.grad)
            input_t.grad.zero_()

            mlu_out = torch.nn.functional.glu(input_t_mlu)
            mlu_out.backward(grad_mlu.as_strided(grad_shape, stride))
            input_grad_mlu = copy.deepcopy(input_t.grad)

            self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_grad_cpu.float(),
                input_grad_mlu.cpu().float(),
                0.003,
                use_MSE=True,
            )


if __name__ == "__main__":
    unittest.main()
