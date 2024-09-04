from __future__ import print_function

import sys
import os
import unittest
import logging
import copy

from itertools import product
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
    def test_take(self):
        shapes = [(2, 3, 24, 30), (2, 3, 33), (2, 24)]
        indices = [
            torch.tensor([0, 1, 2]),
            torch.tensor([10, 20, 40]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([0, 1, 3, 3]),
            torch.LongTensor([[0, 2], [3, 4]]),
            torch.empty((0,), dtype=torch.int64),
        ]
        input_dtypes = [
            torch.float,
            torch.half,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
        for shape, data_type in product(shapes, input_dtypes):
            for idx in indices:
                input = torch.rand(shape, dtype=torch.float).to(data_type)
                out_cpu = torch.take(input, idx)
                out_mlu = torch.take(input.mlu(), idx.mlu())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_take_out(self):
        shapes = [(2, 3, 24, 30), (2, 3, 33), (2, 24)]
        indices = [
            torch.tensor([0, 1, 2]),
            torch.tensor([10, 20, 40]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([0, 1, 3, 3]),
        ]
        input_dtypes = [torch.float, torch.half, torch.uint8, torch.int16, torch.int32]
        for shape, data_type in product(shapes, input_dtypes):
            for idx in indices:
                input = torch.rand(shape, dtype=torch.float).to(data_type)
                out_cpu = torch.zeros_like(idx).to(data_type)
                out_mlu = torch.zeros_like(idx).to(data_type).mlu()
                torch.take(input, idx, out=out_cpu)
                torch.take(input.mlu(), idx.mlu(), out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

        # broadcast case
        input = torch.rand((1, 2, 3, 4), dtype=torch.float)
        index = torch.tensor([[0, 1, 2], [2, 3, 4]])
        output = torch.rand((1, 3), dtype=torch.float)
        output_mlu = output.mlu()
        torch.take(input, index, out=output)
        torch.take(input.mlu(), index.mlu(), out=output_mlu)
        self.assertTensorsEqual(output, output_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_take_not_dense(self):
        shapes = [(2, 32, 24, 30), (2, 32, 33), (24, 24)]
        indices = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 1])]
        input_dtypes = [torch.float, torch.half, torch.uint8, torch.int16, torch.int32]
        for shape, data_type in product(shapes, input_dtypes):
            for idx in indices:
                input = torch.rand(shape, dtype=torch.float).to(data_type)
                out_cpu = torch.take(input[..., 2], idx)
                out_mlu = torch.take(input.mlu()[..., 2], idx.mlu())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_take_channel_last(self):
        shapes = [(2, 3, 24, 30), (2, 3, 33), (2, 24)]
        indices = [
            torch.tensor([0, 1, 2]),
            torch.tensor([10, 20, 40]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([0, 1, 3, 3]),
        ]
        input_dtypes = [torch.float, torch.half, torch.uint8, torch.int16, torch.int32]
        for shape, data_type in product(shapes, input_dtypes):
            for idx in indices:
                input = torch.rand(shape, dtype=torch.float).to(data_type)
                input = self.convert_to_channel_last(input)
                out_cpu = torch.take(input, idx)
                out_mlu = torch.take(input.mlu(), idx.mlu())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_take_negative_idx(self):
        shapes = [(2, 3, 24, 30), (2, 3, 33), (2, 24)]
        indices = [
            torch.tensor([0, -1, -2]),
            torch.tensor([-10, -20, -40]),
            torch.tensor([-1, -2, -3, -4, -5, -6]),
            torch.tensor([0, -1, -3, -3]),
        ]
        input_dtypes = [torch.float, torch.half, torch.uint8, torch.int16, torch.int32]
        for shape, data_type in product(shapes, input_dtypes):
            for idx in indices:
                input = torch.rand(shape, dtype=torch.float).to(data_type)
                out_cpu = torch.take(input, idx)
                out_mlu = torch.take(input.mlu(), idx.mlu())
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_take_empty(self):
        idx = torch.tensor([]).long().to("mlu")
        input = torch.rand((2, 3), dtype=torch.float).to("mlu")
        out_cpu = torch.take(input.cpu(), idx.cpu())
        out_mlu = torch.take(input, idx)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    # This test case is used to test wheter the input is changed
    # @unittest.skip("not test")
    @testinfo()
    def test_take_idx(self):
        input_cpu = torch.randn(3, 4)
        input_mlu = input_cpu.mlu()
        idx_cpu = torch.LongTensor([[0], [-2]])
        idx_mlu = idx_cpu.mlu()
        input_cpu.take(idx_cpu)
        input_mlu.take(idx_mlu)
        self.assertTensorsEqual(idx_cpu.float(), idx_mlu.cpu().float(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_take_backward(self):
        input_cpu = torch.randn((3, 4), dtype=torch.float)
        input_mlu = copy.deepcopy(input_cpu).mlu()
        input_cpu.requires_grad = True
        input_mlu.requires_grad = True
        idx_cpu = torch.LongTensor([[0], [-2]])
        idx_mlu = idx_cpu.mlu()
        out_cpu = torch.take(input_cpu, idx_cpu)
        out_mlu = torch.take(input_mlu, idx_mlu)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        out_grad_cpu = input_cpu.grad
        out_grad_mlu = input_mlu.grad
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_take_exception(self):
        indices = torch.tensor([0, 10, 2]).to("mlu")
        input = torch.rand((2, 3), dtype=torch.float).to("mlu")
        numel = input.numel()
        err_idx = 0
        for idx in indices:
            if idx >= numel or idx < -numel:
                err_idx = idx
                break
        ref_msg = (
            "out of range: tried to access index "
            + str(err_idx.item())
            + " on a tensor of "
            + str(numel)
            + " elements"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.take(input, indices)
        input_complex = torch.rand((2, 3), dtype=torch.complex64).mlu()
        idx = torch.LongTensor([[0], [-2]]).mlu()
        ref_msg = r"\"take\" not implemented for"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.take(input_complex, idx)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_take_bfloat16(self):
        shapes = [(2, 3, 24, 30), (2, 3, 33), (2, 24)]
        indices = [
            torch.tensor([0, 1, 2]),
            torch.tensor([10, 20, 40]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([0, 1, 3, 3]),
            torch.LongTensor([[0, 2], [3, 4]]),
            torch.empty((0,), dtype=torch.int64),
        ]
        input_dtypes = [torch.bfloat16]
        for shape, data_type in product(shapes, input_dtypes):
            for idx in indices:
                input = torch.rand(shape, dtype=data_type)
                input_cpu = input.float()
                input_cpu.requires_grad = True
                input_mlu = self.to_device(input)
                input_mlu.requires_grad = True
                out_cpu = torch.take(input_cpu, idx)
                out_mlu = torch.take(input_mlu, self.to_device(idx))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                # test bfloat16 backward
                grad = torch.randn(out_cpu.shape, dtype=data_type)
                grad_cpu = grad.float()
                grad_mlu = self.to_device(grad)
                out_cpu.backward(grad_cpu)
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    input_cpu.grad.float(),
                    input_mlu.grad.cpu().float(),
                    0.003,
                    use_MSE=True,
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("46GB")
    def test_take_large_bfloat16(self):
        shapes = [(5, 128, 1024, 1024)]
        indices = [
            torch.tensor([0, 1, 2]),
            torch.tensor([10, 20, 40]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([0, 1, 3, 3]),
            torch.LongTensor([[0, 2], [3, 4]]),
            torch.empty((0,), dtype=torch.int64),
        ]
        input_dtypes = [torch.bfloat16]
        for shape, data_type in product(shapes, input_dtypes):
            for idx in indices:
                input = torch.rand(shape, dtype=data_type)
                input_cpu = input.float()
                input_mlu = self.to_device(input)
                out_cpu = torch.take(input_cpu, idx)
                out_mlu = torch.take(input_mlu, self.to_device(idx))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
