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
    run_tests,
    testinfo,
    TestCase,
    skipDtypeNotSupport,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestTraceOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_torch_trace(self):
        dtype_list = [torch.float, torch.half, torch.int, torch.double, torch.long]
        shape_list = [
            (3, 3),
            (5, 8),
            (8, 10),
            (18, 25),
            (36, 20),
            (0, 0),
            (10, 0),
            (0, 3),
        ]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype.is_floating_point:
                    x = torch.randn(shape, dtype=dtype)
                else:
                    x = torch.randint(0, 100, shape, dtype=dtype)
                x_mlu = self.to_mlu(copy.deepcopy(x))
                if dtype == torch.half:
                    x = x.to(torch.float)
                if dtype.is_floating_point:
                    x.requires_grad = True
                    x_mlu.requires_grad = True

                out_cpu = torch.trace(x)
                out_mlu = torch.trace(x_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                if dtype.is_floating_point:
                    grad = torch.rand(out_cpu.shape, dtype=dtype)
                    grad_mlu = self.to_mlu(copy.deepcopy(grad))
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        x.grad, x_mlu.grad.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_trace(self):
        dtype_list = [torch.float, torch.half, torch.int, torch.double, torch.long]
        shape_list = [
            (3, 3),
            (5, 8),
            (8, 10),
            (18, 25),
            (36, 20),
            (0, 0),
            (10, 0),
            (0, 3),
        ]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype.is_floating_point:
                    x = torch.randn(shape, dtype=dtype)
                else:
                    x = torch.randint(0, 100, shape, dtype=dtype)
                x_mlu = self.to_mlu(copy.deepcopy(x))
                if dtype == torch.half:
                    x = x.to(torch.float)
                if dtype.is_floating_point:
                    x.requires_grad = True
                    x_mlu.requires_grad = True

                out_cpu = x.trace()
                out_mlu = x_mlu.trace()
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                if dtype.is_floating_point:
                    grad = torch.rand(out_cpu.shape, dtype=dtype)
                    grad_mlu = self.to_mlu(copy.deepcopy(grad))
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        x.grad, x_mlu.grad.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_torch_trace_not_dense(self):
        dtype_list = [torch.float, torch.half, torch.int, torch.double, torch.long]
        shape_list = [(30, 30), (25, 18), (8, 20), (0, 25)]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype.is_floating_point:
                    x = torch.randn(shape, dtype=dtype)
                else:
                    x = torch.randint(0, 100, shape, dtype=dtype)
                x_mlu = self.to_mlu(copy.deepcopy(x))
                if dtype == torch.half:
                    x = x.to(torch.float)
                x = x[:, :15]
                x_mlu = x_mlu[:, :15]
                if dtype.is_floating_point:
                    x.requires_grad = True
                    x_mlu.requires_grad = True

                out_cpu = torch.trace(x)
                out_mlu = torch.trace(x_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                if dtype.is_floating_point:
                    grad = torch.rand(out_cpu.shape, dtype=dtype)
                    grad_mlu = self.to_mlu(copy.deepcopy(grad))
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        x.grad, x_mlu.grad.cpu().float(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @skipDtypeNotSupport(
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.bool,
        torch.complex32,
        torch.complex64,
        torch.complex128,
    )
    @testinfo()
    def test_trace_unsupported(self, type):
        if type.is_floating_point or type.is_complex:
            input = torch.randn(3, 4, dtype=type).to("mlu")
        else:
            input = torch.randint(0, 2, (3, 4), dtype=type).to("mlu")
        torch.trace(input)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_trace_bfloat16(self):
        input = torch.randn(3, 4)
        out_cpu = torch.trace(input)
        out_mlu = torch.trace(self.to_mlu_dtype(input, torch.bfloat16))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
