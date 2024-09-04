from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import torch
import copy

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


class TestMaskedScatter(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_tensor(self):
        types = [torch.half, torch.float, torch.double, torch.long]
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,), ()]
        err = 0.0
        for t, shape in product(types, shapes):
            x = torch.rand(shape).to(t)
            mask = torch.randn(shape) > 0
            source = torch.rand(shape).to(t)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            source_mlu = self.to_device(source)
            ori_ptr = x_mlu.data_ptr()
            if t == torch.half:
                x, source = x.float(), source.float()
                err = 0.003
            out_cpu = torch.Tensor.masked_scatter_(x, mask, source)
            out_mlu = torch.Tensor.masked_scatter_(x_mlu, mask_mlu, source_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_channels_last_and_not_dense(self):
        shape = (100, 512, 2, 5)

        # channels last
        x = torch.rand(shape, dtype=torch.float)
        mask = torch.randn(shape) > 0
        source = torch.rand(shape, dtype=torch.float)
        x = x.to(memory_format=torch.channels_last)
        mask = mask.to(memory_format=torch.channels_last)
        source = source.to(memory_format=torch.channels_last)
        x_mlu = self.to_device(x)
        mask_mlu = self.to_device(mask)
        source_mlu = self.to_device(source)
        out_cpu = torch.Tensor.masked_scatter_(x, mask, source)
        out_mlu = torch.Tensor.masked_scatter_(x_mlu, mask_mlu, source_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

        # not dense
        x = torch.rand(shape, dtype=torch.float)
        mask = torch.ones(shape, dtype=torch.bool)
        source = torch.rand(shape, dtype=torch.float)
        x_mlu = self.to_device(x)
        mask_mlu = self.to_device(mask)
        source_mlu = self.to_device(source)
        out_cpu = torch.Tensor.masked_scatter_(x[..., 2], mask[..., 2], source[..., 2])
        out_mlu = torch.Tensor.masked_scatter_(
            x_mlu[..., 2], mask_mlu[..., 2], source_mlu[..., 2]
        )
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_backward(self):
        x = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            dtype=torch.float,
            device="cpu",
            requires_grad=True,
        )
        x_mlu = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            dtype=torch.float,
            device="mlu",
            requires_grad=True,
        )
        mask = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, 1]]).bool()
        mask_mlu = mask.mlu()
        source = torch.zeros(3, 3).to(torch.float)
        source_mlu = torch.zeros((3, 3), dtype=torch.float, device="mlu")
        out_cpu = torch.masked_scatter(x, mask, source)
        out_mlu = torch.masked_scatter(x_mlu, mask_mlu, source_mlu)
        grad = torch.randn(3, 3)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        out_grad_cpu = x.grad
        out_grad_mlu = x_mlu.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_exception(self):
        dest = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float, device="mlu"
        )
        mask = torch.tensor(
            (0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device="mlu"
        )
        src = torch.zeros(2, dtype=torch.float, device="mlu")
        ref_msg = (
            r"masked_scatter: expected self and source to have same dtypes but got"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            dest.masked_scatter_(mask, src.double())
        ref_msg = r"masked_scatter: expected BoolTensor or ByteTensor for mask"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            dest.masked_scatter_(mask.int(), src)
        src_legal = torch.zeros(10, device="mlu", dtype=torch.float)
        ref_msg = r"masked_scatter_ received a mask with dtype torch.uint8, "
        with self.assertWarnsRegex(UserWarning, ref_msg):
            dest.masked_scatter_(mask.to(torch.uint8), src_legal)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_exception_2(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)

class TestThatContainsMLUAssertFailure(TestCase):

    def test_masked_scatter_mlu_assert(self):
        user_stream = torch.mlu.Stream()
        with torch.mlu.stream(user_stream):
            dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float, device='mlu')
            mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device='mlu')
            src = torch.zeros(2, dtype=torch.float, device='mlu')
            dest.masked_scatter_(mask, src)
            user_stream.synchronize()

if __name__ == '__main__':
    run_tests()
"""
        )
        # should capture MLU error
        self.assertIn("Device-side assert triggered", stderr)
        # should run only 1 test because it throws unrecoverable error.
        self.assertIn("errors=1", stderr)

    # @unittest.skip("not test")
    @testinfo()
    def test_masked_scatter_with_mix_memory_format(self):
        x = self.convert_to_channel_last(torch.rand(2, 2, 2, 2, dtype=torch.float))
        x_mlu = copy.deepcopy(x).mlu()
        mask = self.convert_to_channel_last(
            torch.tensor(
                [
                    [[[1.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]],
                    [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
                ]
            ).bool()
        )
        source = torch.rand(5, dtype=torch.float)
        torch.ops.aten.masked_scatter_(x, mask, source)
        torch.ops.aten.masked_scatter_(
            x_mlu, self.to_device(mask), self.to_device(source)
        )
        self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_masked_scatter_tensor_bfloat16(self):
        types = [torch.bfloat16]
        shapes = [(100, 512, 2, 5), (100, 512, 2), (100, 512), (100,), ()]
        err = 0.0
        for t, shape in product(types, shapes):
            x = torch.rand(shape, dtype=t)
            mask = torch.randn(shape) > 0
            source = torch.rand(shape, dtype=t)
            x_mlu = self.to_device(x)
            mask_mlu = self.to_device(mask)
            source_mlu = self.to_device(source)
            ori_ptr = x_mlu.data_ptr()
            if t == torch.half:
                x, source = x.float(), source.float()
                err = 0.003
            out_cpu = torch.Tensor.masked_scatter_(x, mask, source)
            out_mlu = torch.Tensor.masked_scatter_(x_mlu, mask_mlu, source_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)
            self.assertTensorsEqual(x, x_mlu.cpu().float(), err, use_MSE=True)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
        # test bfloat16 backward
        shape = (100, 512, 2, 5)
        x = torch.rand(shape, dtype=torch.bfloat16).float()
        mask = torch.randn(shape) > 0
        source = torch.rand(shape, dtype=torch.bfloat16).float()
        x_cpu = torch.nn.Parameter(x)
        x_mlu = torch.nn.Parameter(self.to_mlu_dtype(x, torch.bfloat16))
        out_cpu = torch.masked_scatter(x_cpu, mask, source)
        out_mlu = torch.masked_scatter(
            x_mlu, self.to_device(mask), self.to_mlu_dtype(source, torch.bfloat16)
        )
        grad = torch.randn_like(out_cpu)
        grad_mlu = self.to_mlu_dtype(grad, torch.bfloat16)
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            x_cpu.grad, x_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("24GB")
    def test_masked_scatter_large_bfloat16(self):
        dtype = torch.bfloat16
        shape = (5, 256, 1024, 1024)
        x = torch.randn(shape)
        mask = torch.randn(shape) > 0
        source = torch.randn(shape)
        x_mlu = self.to_mlu_dtype(x, dtype)
        out_cpu = torch.masked_scatter(x, mask, source)
        out_mlu = torch.masked_scatter(
            x_mlu, self.to_device(mask), self.to_mlu_dtype(source, dtype)
        )
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("77GB")
    def test_masked_scatter_large_half(self):
        dtype = torch.half
        shape = (4, 1025, 1024, 1024)
        x = torch.randn(shape)
        mask = torch.randn(shape) > 0
        source = torch.randn(shape)
        x_mlu = self.to_mlu_dtype(x, dtype)
        out_cpu = torch.masked_scatter(x, mask, source)
        out_mlu = torch.masked_scatter(
            x_mlu, self.to_device(mask), self.to_mlu_dtype(source, dtype)
        )
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("77GB")
    def test_masked_scatter_onedim_large_half(self):
        dtype = torch.half
        shape = 4294967297
        x = torch.randn(shape)
        mask = torch.randn(shape) > 0
        source = torch.randn(shape)
        x_mlu = self.to_mlu_dtype(x, dtype)
        out_cpu = torch.masked_scatter(x, mask, source)
        out_mlu = torch.masked_scatter(
            x_mlu, self.to_device(mask), self.to_mlu_dtype(source, dtype)
        )
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
