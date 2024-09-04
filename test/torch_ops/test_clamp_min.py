from __future__ import print_function
import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestClampMinOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_min(self):
        shape_list = [(1, 1, 1, 2), (144, 7, 15, 2), (5), (256, 144)]
        dtype_list = [
            torch.float,
            torch.half,
            torch.double,
            torch.long,
            torch.int,
            torch.int16,
            torch.int8,
        ]  # half is not support for cpu
        min_list = [1, 2]
        product_list = product(shape_list, dtype_list, min_list)
        for (
            shape,
            dtype,
            min_,
        ) in product_list:
            if dtype == torch.half:
                x = torch.randn(shape, dtype=torch.float)
                y = torch.randn(shape, dtype=torch.float)
            else:
                x = torch.randn(shape).to(dtype)
                y = torch.randn(shape).to(dtype)
            out_cpu = torch.clamp_min(
                x,
                min_,
            )
            out_mlu = torch.clamp_min(
                self.to_mlu_dtype(x, dtype),
                min_,
            )
            if dtype is torch.half or dtype is torch.float64:
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test clamp.out
            out_cpu = torch.clamp_min(x, min_, out=y)
            out_mlu = torch.clamp_min(
                self.to_mlu_dtype(x, dtype), min_, out=self.to_mlu_dtype(y, dtype)
            )
            if dtype is torch.half or dtype is torch.float64:
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test inplace operation
            x_cpu = copy.deepcopy(x)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x_cpu.clamp_min_(min_)
            x_mlu.clamp_min_(min_)
            if dtype is torch.half or dtype is torch.float64:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3)
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_min_backward(self):
        for shape in [(2, 3), (8, 224, 224), (1, 3, 16, 16), (1, 3, 16, 16, 3)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)

            out_cpu = torch.clamp_min(x, 0.1)
            out_mlu = torch.clamp_min(self.to_device(x), 0.1)

            grad = torch.randn(out_cpu.shape, dtype=torch.float)

            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)

            x_cpu = copy.deepcopy(x)
            x.grad.zero_()

            outmlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)

            self.assertEqual(outmlu_ptr, out_mlu.data_ptr(), 0)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)
            self.assertTensorsEqual(x, x_cpu, 0)

            # test inplace operation
            x.grad.zero_()

            x_mlu = self.to_device(x)
            x_mlu.clamp_min_(0.1)
            xmlu_ptr = x_mlu.data_ptr()
            x_mlu.backward(self.to_device(grad))
            grad_mlu = copy.deepcopy(x.grad)

            self.assertEqual(xmlu_ptr, x_mlu.data_ptr(), 0)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_min_not_dense(self):
        for shape in [(8, 224, 224, 16), (1, 3, 16, 16), (1, 3, 16, 14)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            input_cpu = x[:, :, :, 3:6]
            out_cpu = torch.clamp_min(input_cpu, 0.1)
            input_mlu = self.to_device(x)
            input_mlu = input_mlu[:, :, :, 3:6]
            out_mlu = torch.clamp_min(input_mlu, 0.1)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_min_channel_last(self):
        for shape in [(8, 224, 224, 16), (1, 3, 16, 16), (1, 3, 16, 14, 25)]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            input_cpu = self.convert_to_channel_last(x)
            out_cpu = torch.clamp_min(input_cpu, 0.1)
            input_mlu = self.to_device(x)
            input_mlu = self.convert_to_channel_last(input_mlu)
            out_mlu = torch.clamp_min(input_mlu, 0.1)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_min_tensor(self):
        shape_list = [((1, 1, 1, 2), (1)), ((1, 2, 3, 4), (1, 2, 3, 4))]
        for shape1, shape2 in shape_list:
            x = torch.randn(shape1, dtype=torch.float)
            y = torch.randn(shape1, dtype=torch.float)
            min_ = torch.randn(shape2, dtype=torch.float)
            out_cpu = torch.clamp_min(
                x,
                min_,
            )
            out_mlu = torch.clamp_min(
                x.to("mlu"),
                min_.to("mlu"),
            )
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test clamp_min_tensor.out
            out_cpu = torch.clamp_min(x, min_, out=y)
            out_mlu = torch.clamp_min(x.to("mlu"), min_.to("mlu"), out=y.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

            # test clamp_min_tensor_
            x_cpu = copy.deepcopy(x)
            x_mlu = x.to("mlu")
            x_cpu.clamp_min_(min_)
            x_mlu.clamp_min_(min_.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_clamp_min_exception(self):
        a = torch.randn(3).to(torch.int).to("mlu")
        b = torch.randn(3).to(torch.int).to("mlu")
        ref_msg = "Found dtype Int but expected Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.clamp_min(a, 0.1, out=b)


if __name__ == "__main__":
    run_tests()
