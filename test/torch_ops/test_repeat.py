from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

# pylint: disable=C0413,C0411
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_repeat(self):
        for in_shape in [
            (1, 2, 2),
            (2, 3, 4, 5),
            (10, 10, 10),
            (4, 5, 3),
            (2, 2),
            (3,),
            (5, 6, 10, 24, 24),
        ]:
            for repeat_size in [(2, 3, 4), (2, 3, 4, 5), (2, 2, 2, 2, 2)]:
                if len(repeat_size) < len(in_shape):
                    continue
                input1 = torch.randn(in_shape, dtype=torch.float)
                output_cpu = input1.repeat(repeat_size)
                output_mlu = self.to_mlu(input1).repeat(repeat_size)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-4, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_complex(self):
        real_part = torch.tensor([1.0, 2.0, 3.0])
        imaginary_part = torch.tensor([0.5, -1.0, 0.0])
        for dtype in [torch.float, torch.double]:
            input = torch.complex(real_part.to(dtype), imaginary_part.to(dtype))
            input_mlu = input.mlu()
            for repeat_size in [(2, 3, 4), (2, 3, 4, 5), (2, 2, 2, 2, 2)]:
                output_cpu = input.repeat(repeat_size)
                output_mlu = input_mlu.repeat(repeat_size)
                self.assertTrue(
                    torch.allclose(output_cpu, output_mlu.cpu(), rtol=1e-5, atol=1e-8)
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_channels_last(self):
        for in_shape in [(2, 3, 4, 5), (1, 2, 2), (2, 2), (3,), (5, 6, 10, 24, 24), ()]:
            for repeat_size in [
                (2, 3, 4),
                (
                    2,
                    3,
                    2,
                    4,
                ),
                (2, 2, 2, 2, 2),
                (),
            ]:
                if len(repeat_size) < len(in_shape):
                    continue
                input1 = torch.randn(in_shape, dtype=torch.float)
                channels_last_input1 = self.convert_to_channel_last(input1)
                output_cpu = channels_last_input1.repeat(repeat_size)
                output_mlu = self.to_mlu(channels_last_input1).repeat(repeat_size)
                output_mlu_channels_first = output_mlu.cpu().float().contiguous()
                self.assertTensorsEqual(
                    output_cpu, output_mlu_channels_first, 3e-4, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_not_dense(self):
        for in_shape in [(10, 10, 10), (4, 5, 3)]:
            for repeat_size in [(2, 3, 4), (2, 3, 4, 5), (2, 2, 2, 2, 2)]:
                if len(repeat_size) < len(in_shape):
                    continue
                input1 = torch.randn(in_shape, dtype=torch.float)
                output_cpu = input1[:, :, :2].repeat(repeat_size)
                output_mlu = self.to_mlu(input1)[:, :, :2].repeat(repeat_size)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-4, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_backward(self):
        N_lst = [4, 16]
        HW_lst = [16, 64]
        C_lst = [2, 8]
        sizes = [3, 5, 3, 5]
        for N in N_lst:
            for HW in HW_lst:
                for C in C_lst:
                    x = torch.randn(N, C, HW, HW, dtype=torch.float, requires_grad=True)
                    out_cpu = x.repeat(sizes)
                    grad = torch.randn(out_cpu.shape, dtype=torch.float)
                    out_cpu.backward(grad)
                    grad_cpu = copy.deepcopy(x.grad)
                    x.grad.zero_()
                    out_mlu = self.to_mlu(x).repeat(sizes)
                    g_m = self.to_mlu(grad)
                    out_mlu.backward(g_m)
                    grad_mlu = copy.deepcopy(x.grad)
                    self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_exception(self):
        a = torch.randn((1, 2, 2)).to("mlu")
        ref_msg = r"Number of dimensions of repeat dims can not be smaller than number"
        ref_msg = ref_msg + " of dimensions of tensor"
        repeat_size = (1, 2)
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.repeat(repeat_size)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_repeat_bfloat16(self):
        for in_shape in [
            (1, 2, 2),
            (2, 3, 4, 5),
            (10, 10, 10),
            (4, 5, 3),
            (2, 2),
            (3,),
            (5, 6, 10, 24, 24),
        ]:
            for repeat_size in [(2, 3, 4), (2, 3, 4, 5), (2, 2, 2, 2, 2)]:
                if len(repeat_size) < len(in_shape):
                    continue
                input = torch.randn(in_shape, dtype=torch.bfloat16)
                output_cpu = input.float().repeat(repeat_size)
                output_mlu = self.to_mlu(input).repeat(repeat_size)
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-4, use_MSE=True
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_repeat_backward_bfloat16(self):
        x = torch.randn((4, 2, 16, 16), dtype=torch.bfloat16)
        size = (3, 5, 3, 5)
        x_cpu = x.float()
        x_cpu.requires_grad = True
        out_cpu = x_cpu.repeat(size)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        out_cpu.backward(grad.float())
        x_mlu = self.to_mlu(x)
        x_mlu.requires_grad = True
        out_mlu = x_mlu.repeat(size)
        out_mlu.backward(self.to_device(grad))
        self.assertTensorsEqual(
            x_cpu.grad, x_mlu.grad.cpu().float(), 0.01, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("46GB")
    def test_repeat_large_bfloat16(self):
        in_shape = (5, 128, 512, 512)
        repeat_size = (2, 2, 1, 2)
        input = torch.randn(in_shape, dtype=torch.bfloat16)
        output_cpu = input.float().repeat(repeat_size)
        output_mlu = self.to_mlu(input).repeat(repeat_size)
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 3e-4, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
