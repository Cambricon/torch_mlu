from __future__ import print_function

import sys
import os
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
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
    skipBFloat16IfNotSupport,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


def generate_tensor_with_dtype(shape, dtype):
    if dtype.is_floating_point:
        cpu_tensor = torch.randn(shape, dtype=dtype).float()
        mlu_tensor = cpu_tensor.to("mlu").to(dtype)
        return cpu_tensor, mlu_tensor
    elif dtype.is_complex:
        cpu_tensor = torch.randn(shape, dtype=dtype)
        mlu_tensor = cpu_tensor.to("mlu")
        return cpu_tensor, mlu_tensor
    elif dtype == torch.bool:
        cpu_tensor = torch.randint(0, 2, shape, dtype=dtype)
        mlu_tensor = cpu_tensor.to("mlu")
        return cpu_tensor, mlu_tensor
    else:
        cpu_tensor = torch.randint(100, shape, dtype=dtype)
        mlu_tensor = cpu_tensor.to("mlu")
        return cpu_tensor, mlu_tensor


class TestFlipOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_flip_torch(self):
        def run_test(x, x_mlu, dims, err):
            out_cpu = torch.flip(x, dims)
            out_mlu = torch.flip(x_mlu, dims)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), err, use_MSE=True)

        type_list = [
            (torch.float, 0),
            (torch.half, 0),
            (torch.int, 0),
            (torch.short, 0),
            (torch.long, 0),
            (torch.uint8, 0),
            (torch.double, 0),
            (torch.int8, 0),
            (torch.bool, 0),
        ]
        for shape in [(1, 3, 2, 2), (2, 30, 80, 80), (64, 3, 224, 224)]:
            for type_, err in type_list:
                for dims in [(2, 0, 1), ()]:
                    x, x_mlu = generate_tensor_with_dtype(shape, type_)
                    # common memory_format, channels_first input
                    run_test(x, x_mlu, dims, err)

                    # channels_last input
                    run_test(
                        x.to(memory_format=torch.channels_last),
                        x_mlu.to(memory_format=torch.channels_last),
                        dims,
                        err,
                    )

                    # not-dense input
                    run_test(x[..., :2], x_mlu[..., :2], dims, err)

    # @unittest.skip("not test")
    @testinfo()
    def test_flip_tensor(self):
        def run_test(x, x_mlu, dims, err):
            out_cpu = x.flip(dims)
            out_mlu = x_mlu.flip(dims)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), err, use_MSE=True)

        type_list = [
            (torch.float, 0),
            (torch.half, 0),
            (torch.int, 0),
            (torch.short, 0),
            (torch.long, 0),
            (torch.uint8, 0),
            (torch.double, 0),
            (torch.int8, 0),
            (torch.bool, 0),
        ]
        for shape in [
            (8, 3, 32, 32),
            (5, 8, 30, 80),
            (30, 6, 3, 20),
            (7, 1, 3, 224),
            (0, 1, 3, 224),
        ]:
            for type_, err in type_list:
                for dims in [(3, 1, 2), ()]:
                    x, x_mlu = generate_tensor_with_dtype(shape, type_)
                    # common memory_format, channels_first input
                    run_test(x, x_mlu, dims, err)

                    # channels_last input
                    run_test(
                        x.to(memory_format=torch.channels_last),
                        x_mlu.to(memory_format=torch.channels_last),
                        dims,
                        err,
                    )

                    # not-dense input
                    run_test(x[..., :2], x_mlu[..., :2], dims, err)

    # @unittest.skip("not test")
    @testinfo()
    def test_flip_backward(self):
        def run_test(x, x_mlu, x_grad, x_grad_mlu, dims):
            out_cpu = x.flip(dims)
            out_mlu = x_mlu.flip(dims)
            out_cpu.backward(x_grad)
            out_grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu.backward(x_grad_mlu)
            out_grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0, use_MSE=True)

        for shape in [
            (8, 3, 32, 32),
            (5, 8, 30, 80),
            (30, 6, 3, 20),
            (7, 1, 3, 224),
            (0, 1, 3, 224),
        ]:
            for dims in [(3, 1, 2), ()]:
                x = torch.randn(shape, requires_grad=True)
                x_mlu = x.to("mlu")
                x_grad = torch.randn(shape)
                x_grad_mlu = x_grad.to("mlu")
                run_test(x, x_mlu, x_grad, x_grad_mlu, dims)

    # @unittest.skip("not test")
    @testinfo()
    def test_flip_is_contiguous(self):
        def run_test(
            x_contiguous, x_mlu_contiguous, dims, memory_format=torch.contiguous_format
        ):
            x_cpu = x_contiguous.transpose(0, 1)
            x_mlu = x_mlu_contiguous.transpose(0, 1)
            out_cpu = x_cpu.flip(dims)
            out_mlu = x_mlu.flip(dims)
            # For non-continuous input, GPU and CPU have different continuous
            # results for output. Consistent MLU and GPU behavior.
            self.assertTrue(out_mlu.is_contiguous(memory_format=memory_format))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=True)

        for shape in [(8, 3, 32, 32), (5, 8, 30, 80), (30, 6, 3, 20)]:
            for dims in [(3, 1, 2), ()]:
                x = torch.randn(shape)
                x_mlu = x.to("mlu")
                run_test(x, x_mlu, dims)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("58GB")
    def test_flip_large(self):
        def run_test(x, x_mlu, dims, err):
            out_cpu = torch.flip(x, dims)
            out_mlu = torch.flip(x_mlu, dims)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), err, use_MSE=True)

        type_list = [(torch.half, 0)]
        for shape in [(5, 1024, 1024, 1024)]:
            for type_, err in type_list:
                for dims in [(2, 0, 1), ()]:
                    x, x_mlu = generate_tensor_with_dtype(shape, type_)
                    # common memory_format, channels_first input
                    run_test(x, x_mlu, dims, err)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_flip_bfloat16(self):
        shape = [8, 3, 32, 32]
        dims = [3, 1, 2]
        x = torch.randn(shape, requires_grad=True, dtype=torch.bfloat16)
        x_mlu = x.to("mlu")
        x_grad = torch.randn(shape, dtype=torch.bfloat16)
        x_grad_mlu = x_grad.to("mlu")
        out_cpu = x.flip(dims)
        out_mlu = x_mlu.flip(dims)
        out_cpu.backward(x_grad)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(x_grad_mlu)
        out_grad_mlu = copy.deepcopy(x.grad)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=True)
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0, use_MSE=True)


if __name__ == "__main__":
    run_tests()
