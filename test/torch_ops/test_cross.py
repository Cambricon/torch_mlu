import sys
import os
import math
import unittest
import logging
import copy

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    skipDtypeNotSupport,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)

logging.basicConfig(level=logging.DEBUG)


class TestCrossOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_cross(self):
        shape_list = [(3,), (3, 4, 4), (10, 3), (4, 5, 3, 3), (10, 3, 3, 3, 5, 6)]
        dtype_list = [torch.float, torch.half, torch.int32, torch.double, torch.int64]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype.is_floating_point:
                    a = torch.rand(shape, dtype=torch.float).to(dtype)
                    b = torch.rand(shape, dtype=torch.float).to(dtype)
                else:
                    a = torch.randint(0, 100, shape, dtype=dtype)
                    b = torch.randint(0, 100, shape, dtype=dtype)
                a_mlu = copy.deepcopy(a).to("mlu")
                b_mlu = copy.deepcopy(b).to("mlu")
                if dtype == torch.half:
                    a = a.to(torch.float)
                    b = b.to(torch.float)
                if dtype.is_floating_point:
                    a.requires_grad = True
                    b.requires_grad = True
                    a_mlu.requires_grad = True
                    b_mlu.requires_grad = True

                out_cpu = torch.cross(a, b)
                out_mlu = torch.cross(a_mlu, b_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

                if dtype.is_floating_point:
                    grad = torch.rand(shape, dtype=torch.float).to(dtype)
                    grad_mlu = copy.deepcopy(grad).to("mlu")
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        a.grad, a_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        b.grad, b_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_PYTORCH_11152(self):
        shape = (1, 3, 1, 4, 4)
        dtype = torch.float
        a = torch.rand(shape, dtype=torch.float).to(dtype)
        b = torch.rand(shape, dtype=torch.float).to(dtype)
        a.as_strided_(a.size(), stride=(4, 1, 4, 12, 3))
        b.as_strided_(b.size(), stride=(48, 1, 48, 12, 3))
        a_mlu = copy.deepcopy(a).to("mlu")
        b_mlu = copy.deepcopy(b).to("mlu")
        a.requires_grad = True
        b.requires_grad = True
        a_mlu.requires_grad = True
        b_mlu.requires_grad = True

        out_cpu = torch.cross(a, b)
        out_mlu = torch.cross(a_mlu, b_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

        grad = torch.rand_like(out_cpu)
        grad_mlu = copy.deepcopy(grad).to("mlu")
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(a.grad, a_mlu.grad.cpu().float(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(b.grad, b_mlu.grad.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_channelslast(self):
        shape_list = [
            (5, 3, 3, 3),
            (2, 4, 3, 2),
            (3, 4, 5, 6),
            (4, 5, 6, 3),
            (0, 3, 5, 6),
            (2, 3, 2, 5, 6),
            (3, 5, 6, 7, 8),
            (4, 5, 3, 6, 6),
            (4, 5, 6, 3, 7),
            (4, 5, 6, 7, 3),
            (4, 3, 9, 0, 7),
        ]
        for shape in shape_list:
            a = torch.rand(shape, dtype=torch.float)
            b = torch.rand(shape, dtype=torch.float)
            a = self.convert_to_channel_last(a)
            b = self.convert_to_channel_last(b)
            a_mlu = copy.deepcopy(a).to("mlu")
            b_mlu = copy.deepcopy(b).to("mlu")
            a.requires_grad = True
            b.requires_grad = True
            a_mlu.requires_grad = True
            b_mlu.requires_grad = True

            out_cpu = torch.cross(a, b)
            out_mlu = torch.cross(a_mlu, b_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

            grad = torch.rand(shape, dtype=torch.float)
            grad = self.convert_to_channel_last(grad)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                a.grad, a_mlu.grad.cpu().float(), 3e-3, use_MSE=True
            )
            self.assertTensorsEqual(
                b.grad, b_mlu.grad.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_empty_input(self):
        shape_list = [(3, 4, 0), (0, 3), (4, 0, 3, 3), (10, 3, 3, 3, 5, 0)]
        dtype_list = [torch.float, torch.half, torch.int32]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype.is_floating_point:
                    a = torch.rand(shape, dtype=torch.float).to(dtype)
                    b = torch.rand(shape, dtype=torch.float).to(dtype)
                else:
                    a = torch.randint(0, 100, shape, dtype=dtype)
                    b = torch.randint(0, 100, shape, dtype=dtype)
                a_mlu = copy.deepcopy(a).to("mlu")
                b_mlu = copy.deepcopy(b).to("mlu")
                if dtype == torch.half:
                    a = a.to(torch.float)
                    b = b.to(torch.float)
                if dtype.is_floating_point:
                    a.requires_grad = True
                    b.requires_grad = True
                    a_mlu.requires_grad = True
                    b_mlu.requires_grad = True

                out_cpu = torch.cross(a, b)
                out_mlu = torch.cross(a_mlu, b_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=False)

                if dtype.is_floating_point:
                    grad = torch.rand(shape, dtype=torch.float).to(dtype)
                    grad_mlu = copy.deepcopy(grad).to("mlu")
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        a.grad, a_mlu.grad.cpu().float(), 0, use_MSE=False
                    )
                    self.assertTensorsEqual(
                        b.grad, b_mlu.grad.cpu().float(), 0, use_MSE=False
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_not_dense(self):
        input = torch.rand(4, 3).to(torch.float)
        value = math.nan
        a_cpu = input.new_empty(input.shape + (2,))
        a_cpu[..., 0] = value
        a_cpu[..., 1] = input.detach()
        a_cpu = a_cpu[..., 1]
        input_mlu = input.mlu()
        a_mlu = input_mlu.new_empty(input_mlu.shape + (2,))
        a_mlu[..., 0] = value
        a_mlu[..., 1] = input_mlu.detach()
        a_mlu = a_mlu[..., 1]

        input = torch.rand(4, 3).to(torch.float)
        value = math.nan
        b_cpu = input.new_empty(input.shape + (2,))
        b_cpu[..., 0] = value
        b_cpu[..., 1] = input.detach()
        b_cpu = b_cpu[..., 1]
        input_mlu = input.mlu()
        b_mlu = input_mlu.new_empty(input_mlu.shape + (2,))
        b_mlu[..., 0] = value
        b_mlu[..., 1] = input_mlu.detach()
        b_mlu = b_mlu[..., 1]

        out_cpu = torch.cross(a_cpu, b_cpu)
        out_mlu = torch.cross(a_mlu, b_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_out(self):
        a = torch.rand(4, 3).to(torch.float)
        b = torch.rand(4, 3).to(torch.float)
        out_cpu = torch.zeros(4, 3).to(torch.float)
        a_mlu = a.mlu()
        b_mlu = b.mlu()
        out_mlu = torch.zeros(4, 3).to(torch.float).mlu()
        torch.cross(a, b, out=out_cpu)
        torch.cross(a_mlu, b_mlu, out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_out_non_dense(self):
        a = torch.rand(2, 4, 3).to(torch.float)
        b = torch.rand(2, 4, 3).to(torch.float)
        out_cpu = torch.zeros(2, 5, 3).to(torch.float)
        a_mlu = a.mlu()
        b_mlu = b.mlu()
        out_mlu = torch.zeros(2, 5, 3).to(torch.float).mlu()
        torch.cross(a, b, out=out_cpu[:, 1:, :])
        torch.cross(a_mlu, b_mlu, out=out_mlu[:, 1:, :])
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_set_dim(self):
        shape = (5, 2, 4, 3)
        a = torch.rand(shape).to(torch.float)
        b = torch.rand(shape).to(torch.float)
        out_cpu = torch.zeros(shape).to(torch.float)
        a_mlu = a.mlu()
        b_mlu = b.mlu()
        out_mlu = torch.zeros(shape).to(torch.float).mlu()
        torch.cross(a, b, dim=3, out=out_cpu)
        torch.cross(a_mlu, b_mlu, dim=3, out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_linalg_cross(self):
        shape_list = [(3,), (4, 4, 3), (10, 3), (4, 5, 3, 3), (10, 6, 3, 3, 5, 3)]
        dtype_list = [torch.float, torch.half, torch.int32]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype.is_floating_point:
                    a = torch.rand(shape, dtype=torch.float).to(dtype)
                    b = torch.rand(shape, dtype=torch.float).to(dtype)
                else:
                    a = torch.randint(0, 100, shape, dtype=dtype)
                    b = torch.randint(0, 100, shape, dtype=dtype)
                a_mlu = copy.deepcopy(a).to("mlu")
                b_mlu = copy.deepcopy(b).to("mlu")
                if dtype == torch.half:
                    a = a.to(torch.float)
                    b = b.to(torch.float)
                if dtype.is_floating_point:
                    a.requires_grad = True
                    b.requires_grad = True
                    a_mlu.requires_grad = True
                    b_mlu.requires_grad = True

                out_cpu = torch.linalg.cross(a, b)
                out_mlu = torch.linalg.cross(a_mlu, b_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

                if dtype.is_floating_point:
                    grad = torch.rand(shape, dtype=torch.float).to(dtype)
                    grad_mlu = copy.deepcopy(grad).to("mlu")
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        a.grad, a_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        b.grad, b_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_cross(self):
        shape_list = [(3,), (3, 4, 5), (10, 3), (4, 5, 3, 3), (10, 3, 3, 3, 5, 6)]
        dtype_list = [torch.float, torch.half, torch.int32]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype.is_floating_point:
                    a = torch.rand(shape, dtype=torch.float).to(dtype)
                    b = torch.rand(shape, dtype=torch.float).to(dtype)
                else:
                    a = torch.randint(0, 100, shape, dtype=dtype)
                    b = torch.randint(0, 100, shape, dtype=dtype)
                a_mlu = copy.deepcopy(a).to("mlu")
                b_mlu = copy.deepcopy(b).to("mlu")
                if dtype == torch.half:
                    a = a.to(torch.float)
                    b = b.to(torch.float)
                if dtype.is_floating_point:
                    a.requires_grad = True
                    b.requires_grad = True
                    a_mlu.requires_grad = True
                    b_mlu.requires_grad = True

                out_cpu = a.cross(b)
                out_mlu = a_mlu.cross(b_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

                if dtype.is_floating_point:
                    grad = torch.rand(shape, dtype=torch.float).to(dtype)
                    grad_mlu = copy.deepcopy(grad).to("mlu")
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        a.grad, a_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        b.grad, b_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_permute(self):
        shape_list = [(2, 3)]
        dtype_list = [torch.float]
        for dtype in dtype_list:
            for shape in shape_list:
                a = torch.rand(shape, dtype=torch.float).to(dtype).permute((1, 0))
                b = torch.rand(shape, dtype=torch.float).to(dtype).permute((1, 0))
                a_mlu = copy.deepcopy(a).to("mlu")
                b_mlu = copy.deepcopy(b).to("mlu")
                if dtype.is_floating_point:
                    a.requires_grad = True
                    b.requires_grad = True
                    a_mlu.requires_grad = True
                    b_mlu.requires_grad = True

                out_cpu = torch.cross(a, b)
                out_mlu = torch.cross(a_mlu, b_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

                if dtype.is_floating_point:
                    grad = torch.rand(out_cpu.shape, dtype=torch.float).to(dtype)
                    grad_mlu = copy.deepcopy(grad).to("mlu")
                    if dtype == torch.half:
                        grad = grad.to(torch.float)
                    out_cpu.backward(grad)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(
                        a.grad, a_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        b.grad, b_mlu.grad.cpu().float(), 3e-3, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_cross_exception(self):
        a = torch.randn((2, 3, 4, 4), dtype=torch.float).to("mlu")
        b = torch.randn((2, 3, 4, 4), dtype=torch.half).to("mlu")
        ref_msg = r"Found dtype Half but expected Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = torch.cross(a, b)

    # @unittest.skip("not test")
    @skipDtypeNotSupport(
        torch.int8, torch.uint8, torch.int16, torch.complex64, torch.complex128
    )
    @testinfo()
    def test_cross_unsupported(self, type):
        if type.is_floating_point or type.is_complex:
            a = torch.randn((2, 3, 4, 4), dtype=type).to("mlu")
            b = torch.randn((2, 3, 4, 4), dtype=type).to("mlu")
        else:
            a = torch.randint(0, 2, (2, 3, 4, 4), dtype=type).to("mlu")
            b = torch.randint(0, 2, (2, 3, 4, 4), dtype=type).to("mlu")
        output = torch.cross(a, b)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_cross_bfloat16(self, type):
        a = torch.randn((2, 3, 4, 4))
        b = torch.randn((2, 3, 4, 4))
        out_cpu = torch.cross(a, b)
        out_mlu = torch.cross(
            self.to_mlu_dtype(a, torch.bfloat16), self.to_mlu_dtype(b, torch.bfloat16)
        )
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    run_tests()
