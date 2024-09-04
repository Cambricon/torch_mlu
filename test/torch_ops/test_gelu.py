from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
import torch.nn.functional as F
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    read_card_info,
    largeTensorTest,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestGeluOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_gelu_backward(self):
        def run_test(x_0, approximate="none"):
            x_0.requires_grad = True
            x_mlu = x_0.to("mlu")

            out_cpu = F.gelu(x_0, approximate=approximate)
            out_mlu = F.gelu(x_mlu, approximate=approximate)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.to("mlu")

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x_0.grad)
            x_0.grad.zero_()
            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x_0.grad)

            # cnnl gelu kernel has precision error 0.0005, when the max value
            # of input is smaller, the diff of mse will be bigger.
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.06, use_MSE=True
            )
            self.assertTensorsEqual(
                out_grad_cpu.float(), out_grad_mlu.cpu().float(), 0.06, use_MSE=True
            )

        in_shape = [
            (50),
            (35, 46),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (16, 17, 18, 19, 20, 21),
        ]
        type_list = [torch.float, torch.half]
        type_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        for shape in in_shape:
            for typeId in type_list:
                for approximate in ["none", "tanh"]:
                    x_0 = torch.randn(shape, dtype=typeId, requires_grad=True)
                    run_test(x_0.detach(), approximate=approximate)

                    # channels_last input
                    if x_0.dim() == 4:
                        run_test(
                            x_0.to(memory_format=torch.channels_last).detach(),
                            approximate=approximate,
                        )

                    # not-dense input
                    run_test(x_0[..., :2].detach(), approximate=approximate)

    # @unittest.skip("not test")
    @testinfo()
    def test_gelu_permute(self):
        import random

        for in_shape in [
            (8, 224, 224),
            (1, 3, 16, 16, 4),
            (1, 3, 16, 16, 3, 6),
            (1, 3, 16, 16, 4, 15, 8),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            size = np.arange(len(in_shape))
            random.shuffle(size)
            input_ = torch.permute(input_, tuple(size))
            input_cpu = copy.deepcopy(input_)
            output_cpu = F.gelu(input_)
            output_mlu = F.gelu(self.to_mlu(input_cpu))
            # cnnl gelu kernel has precision error 0.0005, when the max value
            # of input is smaller, the diff of mse will be bigger.
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.06, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gelu_backwark_permute(self):
        input_cpu = torch.randn((4, 3, 2, 1), dtype=torch.float, requires_grad=True)
        out_cpu = F.gelu(input_cpu)
        out_mlu = F.gelu(input_cpu.to("mlu"))
        grad = torch.randn((4, 3, 1, 2), dtype=torch.float)  # test backward
        out_cpu.backward(torch.permute(grad, (0, 1, 3, 2)))
        grad_cpu = copy.deepcopy(input_cpu.grad)
        input_cpu.grad.zero_()

        out_mlu.backward(grad.to("mlu").permute(0, 1, 3, 2))
        grad_mlu = input_cpu.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("70GB")
    def test_gelu_large(self):
        in_shape = [(5, 1024, 1024, 1024)]
        type_list = [torch.half]
        for shape in in_shape:
            for typeId in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(typeId)
                x_mlu = x.to("mlu")

                # use float on cpu kernel
                out_cpu = F.gelu(x_0)
                out_mlu = F.gelu(x_mlu)

                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.to("mlu")

                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)

                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_grad_cpu, out_grad_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
