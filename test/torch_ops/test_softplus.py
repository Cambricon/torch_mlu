from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import numpy as np

import torch
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)

logging.basicConfig(level=logging.DEBUG)


class TestSoftplusOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_softplus(self):
        for in_shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            output_cpu = F.softplus(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.softplus(self.to_device(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_permute(self):
        import random

        for in_shape in [
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16, 4),
            (1, 3, 16, 16, 3, 6),
            (1, 3, 16, 16, 4, 15, 8),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            size = np.arange(len(in_shape))
            random.shuffle(size)
            input_mlu = input_.to("mlu")
            input_ = torch.permute(input_, tuple(size))
            input_mlu = torch.permute(input_mlu, tuple(size))
            output_cpu = F.softplus(input_)
            output_mlu = F.softplus(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_channels_last(self):
        for in_shape in [(2, 3, 24, 30), (1, 1, 1, 30)]:
            input_ = torch.randn(in_shape, dtype=torch.float).to(
                memory_format=torch.channels_last
            )
            output_cpu = F.softplus(input_)
            input_cpu = copy.deepcopy(input_)
            output_mlu = F.softplus(self.to_device(input_))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_not_dense(self):
        for in_shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            for con in [True, False]:
                input_ = torch.randn(in_shape, dtype=torch.float)
                if con is True:
                    input_ = self.get_not_contiguous_tensor(input_)
                output_cpu = F.softplus(input_)
                input_cpu = copy.deepcopy(input_)
                output_mlu = F.softplus(self.to_device(input_))
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_dtype(self):
        for in_shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            dtypes = [torch.float, torch.double]
            for dtype in dtypes:
                input_ = torch.randn(in_shape).to(dtype)
                output_cpu = F.softplus(input_)
                input_cpu = copy.deepcopy(input_)
                output_mlu = F.softplus(self.to_device(input_))
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(input_cpu, input_, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_boundary_value(self):
        for number in [0, 0.0001, -0.0001, 999999999]:
            x = torch.tensor(number, dtype=torch.float)
            output_cpu = F.softplus(x)
            input_cpu = copy.deepcopy(x)
            output_mlu = F.softplus(self.to_device(x))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, x, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_softplus_backward(self):
        for shape in [(1), (2, 3), (8, 24, 24), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = F.softplus(x, beta=1, threshold=20)
            out_mlu = F.softplus(self.to_device(x), beta=1, threshold=20)
            out_mlu_ptr = out_mlu.data_ptr()
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x_cpu = copy.deepcopy(x)
            x.grad.zero_()
            out_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(x, x_cpu, 0)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_softplus_large(self):
        data_type = torch.half
        for in_shape in [(5, 1024, 1024, 1024)]:
            input = torch.randn(in_shape, dtype=torch.float)
            output_cpu = F.softplus(input)
            output_mlu = F.softplus(self.to_mlu_dtype(input, data_type))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_softplus_bfloat16(self):
        in_shape = [5, 3, 4, 2]
        input = torch.randn(in_shape, dtype=torch.bfloat16)
        output_cpu = F.softplus(input)
        output_mlu = F.softplus(self.to_mlu(input))
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
