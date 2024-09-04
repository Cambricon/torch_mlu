from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    read_card_info,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestDiagOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_diag(self):
        inputs_list = [
            ((3, 3), -1),
            ((55, 82), 30),
            ((88,), -2),
            ((0,), 3),
            ((4, 0), -3),
        ]
        type_list = [
            torch.uint8,
            torch.int8,
            torch.float32,
            torch.int16,
            torch.int32,
            torch.double,
            torch.long,
        ]
        for shape, diagonal in inputs_list:
            for dtype in type_list:
                x = torch.randint(-10, 10, shape).type(dtype)
                out_cpu = torch.diag(x, diagonal)
                out_mlu = torch.diag(self.to_device(x), diagonal)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_diag_no_dense(self):
        inputs_list = [
            ((3, 3), -1),
            ((55, 82), 30),
            ((88,), -2),
            ((0,), 3),
            ((4, 0), -3),
        ]
        type_list = [
            torch.uint8,
            torch.int8,
            torch.float32,
            torch.int16,
            torch.int32,
            torch.double,
            torch.long,
        ]
        for shape, diagonal in inputs_list:
            for dtype in type_list:
                x = torch.randint(-10, 10, shape).type(dtype)
                x_mlu = self.to_mlu(x)[..., ::2]
                x = x[..., ::2]
                out_cpu = torch.diag(x, diagonal)
                out_mlu = torch.diag(x_mlu, diagonal)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_diag_out(self):
        inputs_list = [((5, 8), 3)]
        for shape, diagonal in inputs_list:
            x = torch.randn(shape)
            out_cpu = torch.randn(5)
            out_mlu = self.to_device(torch.randn(5))
            origin_ptr = out_mlu.data_ptr()
            torch.diag(x, diagonal=diagonal, out=out_cpu)
            torch.diag(self.to_device(x), diagonal=diagonal, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertEqual(origin_ptr, out_mlu.data_ptr())
            out_cpu = torch.randn(2)
            out_mlu = self.to_device(torch.randn(2))
            torch.diag(x, diagonal=diagonal, out=out_cpu)
            torch.diag(self.to_device(x), diagonal=diagonal, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_diag_backward(self):
        x = torch.randn((5, 8), requires_grad=True)
        x_mlu = x.mlu()
        diagnal = -1
        out_cpu = torch.diag(x, diagnal)
        out_mlu = torch.diag(x_mlu, diagnal)
        grad = torch.randn(4)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

        x = torch.randn((5), requires_grad=True)
        x_mlu = x.mlu()
        diagnal = 0
        out_cpu = torch.diag(x, diagnal)
        out_mlu = torch.diag(x_mlu, diagnal)
        grad = torch.randn(5, 5)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_diag_embed(self):
        a = torch.randn((10, 10))
        a_cpu = torch.nn.Parameter(a)
        a_mlu = torch.nn.Parameter(a.mlu())
        result_cpu = torch.diag_embed(a_cpu)
        result_mlu = torch.diag_embed(a_mlu)
        result_cpu.sum().backward()
        result_mlu.sum().backward()
        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(a_cpu.grad, a_mlu.grad.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_diag_exception(self):
        shape1 = (5, 8, 3)
        x = self.to_device(torch.randn(shape1))
        msg = "diag\(\): Supports 1D or 2D tensors. Got 3D"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.diag(x, diagonal=1)
        shape = (5, 8)
        x_mlu = self.to_device(torch.randn(shape))
        out_mlu = self.to_device(torch.randn(shape).int())
        msg1 = r"The datatype of out"
        with self.assertRaisesRegex(RuntimeError, msg1):
            torch.diag(x_mlu, diagonal=3, out=out_mlu)

        shape = (5, 8)
        x_mlu = self.to_device(torch.randn(shape))
        out_mlu = self.to_device(torch.randn(shape))
        # TODO(chentianyi1): change msg2 after all required atens are successfully adapted
        # msg2 = "k must in row and col range of input"
        msg2 = "numel: integer multiplication overflow"
        with self.assertRaisesRegex(RuntimeError, msg2):
            torch.diag(x_mlu, diagonal=10, out=out_mlu)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_diag_bfloat16(self):
        shape = [3, 3]
        diagonal = 1
        dtype = torch.bfloat16
        x = torch.randint(-10, 10, shape).type(dtype)
        out_cpu = torch.zeros(2, dtype=dtype)
        out_mlu = torch.zeros(2, dtype=dtype, device="mlu")
        torch.diag(x, diagonal, out=out_cpu)
        torch.diag(self.to_device(x), diagonal, out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("21GB")
    def test_diag_large(self):
        shape = [65537, 65537]
        diagonal = 1
        dtype = torch.float
        x = torch.randint(-10, 10, shape).type(dtype)
        out_cpu = torch.diag(x, diagonal)
        out_mlu = torch.diag(self.to_device(x), diagonal)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)


if __name__ == "__main__":
    run_tests()
