from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
import torch.nn.functional as F
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestSiluOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_silu(self):
        def run_test(x_0, dtype, if_inplace=False):
            x = x_0.to(dtype)
            x_mlu = x.mlu()

            out_cpu = F.silu(x_0, inplace=if_inplace)
            out_mlu = F.silu(x_mlu, inplace=if_inplace)

            self.assertTensorsEqual(
                out_cpu,
                out_mlu.cpu(),
                0.003,
                use_MSE=True,
            )

        in_shape = [
            (50),
            (35, 46),
            (16, 27, 38),
            (128, 4, 128, 124),
            (2, 3, 4, 5, 6),
            (0,),
        ]
        type_list = [torch.float, torch.half]
        # inplace
        if_inplace_list = [True, False]
        for shape, typeId, if_inplace in product(in_shape, type_list, if_inplace_list):
            x_0 = torch.randn(shape, dtype=typeId, requires_grad=False)
            run_test(x_0, typeId, if_inplace)

            # channels_last input
            if x_0.dim() == 4:
                run_test(x_0.to(memory_format=torch.channels_last), typeId, if_inplace)

            # not-dense input
            run_test(self.get_not_contiguous_tensor(x_0), typeId, if_inplace)

    # @unittest.skip("not test")
    @testinfo()
    def test_silu_backward(self):
        def run_test(x_0, dtype):
            x_0.requires_grad = True
            x = x_0.to(dtype)
            x_mlu = x.mlu()

            out_cpu = F.silu(x_0)
            out_mlu = F.silu(x_mlu)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.mlu()

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x_0.grad)
            x_0.grad.zero_()
            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x_0.grad)

            self.assertTensorsEqual(
                out_grad_cpu,
                out_grad_mlu.cpu(),
                0.003,
                use_MSE=True,
            )

        in_shape = [
            (50),
            (35, 46),
            (16, 27, 38),
            (128, 4, 128, 124),
            (2, 3, 4, 5, 6),
            (0,),
        ]
        type_list = [torch.float, torch.half]
        for shape, typeId in product(in_shape, type_list):
            x_0 = torch.randn(shape, dtype=typeId, requires_grad=False)
            run_test(x_0.detach(), typeId)

            # channels_last input
            if x_0.dim() == 4:
                run_test(x_0.to(memory_format=torch.channels_last).detach(), typeId)

            # not-dense input
            run_test(self.get_not_contiguous_tensor(x_0).detach(), typeId)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_silu_bfloat16(self):
        x = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16)
        x.requires_grad = True
        x_mlu = x.mlu()
        out_cpu = F.silu(x)
        out_mlu = F.silu(x_mlu)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = copy.deepcopy(x.grad)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.005, use_MSE=True)
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.005, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
