from __future__ import print_function

import sys
import os
import unittest
import logging
import copy

from itertools import product
import numpy as np
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
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)
# If accumulate is false and duplicate indices exist, which means there are
# several values to be inserted into the same location, the result will be
# non-deterministic because the order of values to be inserted is not guaranteed.
# Thus, only non-duplicated indices will be testes in this series of unit tests.
# In addition, When accumulate is true, the type of indices is int32, and the type
# of values is half or float, the precision may be poor if the same location is added
# up too many times on mlu-200 series. Therefore, only small-size indices will be tested.


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    # since the kernel does not support data overflow when accumulate=true,
    # we only use small integers and floating numbers here to check whether
    # the aten returns correct result when there is no risk of overflow.
    def test_put_(self):
        shapes = [(2, 32, 24, 30), (2, 32, 33), (24, 24)]
        indices_shapes = [(2, 2), (2, 3), (1, 3)]
        input_dtypes = [
            torch.float,
            torch.half,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.long,
        ]
        accumulate_list = [True, False]
        for items in product(shapes, input_dtypes, indices_shapes, accumulate_list):
            # items[0]: input shape
            # items[1]: input data type
            # items[2]: indices shape
            # items[3]: whether to accumulate the source to input
            input = torch.randn(items[0], dtype=torch.float)
            idx = torch.linspace(0, input.numel() - 1, np.prod(items[2]))
            idx = idx.to(torch.long).reshape(items[2])
            source = torch.rand_like(idx, dtype=torch.float)
            if items[1] not in (torch.float, torch.half):
                source = torch.randint(0, 50, items[2]).to(items[1])
                input = torch.randint(0, 50, items[0]).to(items[1])
            else:
                input = input.to(items[1])
                source = source.to(items[1])
            input_mlu = input.mlu()
            out_cpu = input.put_(idx, source, items[3])
            out_mlu = input_mlu.put_(idx.mlu(), source.mlu(), items[3])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_put__not_dense(self):
        shapes = [(2, 32, 24, 30), (2, 32, 33, 28), (3, 7, 6, 24, 24)]
        indices_shapes = [(2, 2), (2, 3), (1, 3)]
        input_dtypes = [torch.float, torch.half]
        accumulate_list = [True, False]
        for items in product(shapes, input_dtypes, indices_shapes, accumulate_list):
            # items[0]: input shape
            # items[1]: input data type
            # items[2]: indices shape
            # items[3]: whether to accumulate the source to input
            input = torch.rand(items[0], dtype=torch.float).to(items[1])
            input_mlu = input.mlu()
            idx = torch.linspace(
                -input[..., 2].numel(), input[..., 2].numel() - 1, np.prod(items[2])
            )
            idx = idx.to(torch.long).reshape(items[2])
            source = torch.rand_like(idx, dtype=torch.float).to(items[1])
            out_cpu = input[..., 2].put_(idx, source, items[3])
            out_mlu = input_mlu[..., 2].put_(idx.mlu(), source.mlu(), items[3])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_put__channel_last(self):
        shapes = [(2, 32, 24, 30), (2, 32, 33, 28), (3, 7, 6, 24, 24)]
        indices_shapes = [(2, 2), (2, 3), (1, 3)]
        input_dtypes = [torch.float, torch.half]
        accumulate_list = [True, False]
        for items in product(shapes, input_dtypes, indices_shapes, accumulate_list):
            # items[0]: input shape
            # items[1]: input data type
            # items[2]: indices shape
            # items[3]: whether to accumulate the source to input
            input = torch.rand(items[0], dtype=torch.float).to(items[1])
            input_mlu = input.mlu()
            input_mlu = self.convert_to_channel_last(input_mlu)
            idx = torch.linspace(
                -input[..., 2].numel(), input[..., 2].numel() - 1, np.prod(items[2])
            )
            idx = idx.to(torch.long).reshape(items[2])
            source = torch.rand_like(idx, dtype=torch.float).to(items[1])
            out_cpu = input.put_(idx, source, items[3])
            out_mlu = input_mlu.put_(idx.mlu(), source.mlu(), items[3])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_put__empty(self):
        idx = torch.tensor([]).long()
        input = torch.rand((2, 3), dtype=torch.float)
        source = torch.rand_like(idx, dtype=torch.float)
        out_cpu = input.put_(idx, source, True)
        out_mlu = input.mlu().put_(idx.mlu(), source.mlu(), True)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_put__idx(self):
        input_cpu = torch.randn(2, 5, 3)
        input_mlu = input_cpu.mlu()
        idx_cpu = torch.LongTensor([[0], [-2]])
        idx_mlu = idx_cpu.mlu()
        tensor_cpu = torch.LongTensor([[3], [4]]).to(dtype=torch.float)
        tensor_mlu = tensor_cpu.mlu()
        input_cpu.put_(idx_cpu, tensor_cpu)
        input_mlu.put_(idx_mlu, tensor_mlu)
        self.assertTensorsEqual(idx_cpu.float(), idx_mlu.cpu().float(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_put__backward(self):
        input_cpu = torch.randn((2, 5, 3), dtype=torch.float)
        input_mlu = copy.deepcopy(input_cpu).mlu()
        input_cpu.requires_grad = True
        input_mlu.requires_grad = True
        idx_cpu = torch.LongTensor([[0], [-2]])
        idx_mlu = idx_cpu.mlu()
        tensor_cpu = torch.LongTensor([[3], [4]]).to(dtype=torch.float)
        tensor_mlu = tensor_cpu.mlu()
        out_cpu = torch.Tensor.put(input_cpu, idx_cpu, tensor_cpu)
        out_mlu = torch.Tensor.put(input_mlu, idx_mlu, tensor_mlu)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(input_cpu.grad, input_mlu.grad.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_put__exception(self):
        idx = torch.tensor([0, 10, 2])
        input = torch.rand((2, 3), dtype=torch.float)
        input_mlu = input.mlu()
        source = torch.rand_like(idx, dtype=torch.float)
        with self.assertRaises(IndexError) as info_cpu:
            input.put_(idx, source, True)
        with self.assertRaises(RuntimeError) as info_mlu:
            input_mlu.put_(idx.mlu(), source.mlu(), True)
        self.assertEqual(info_cpu.exception.args[0], info_mlu.exception.args[0])

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_put__bfloat16(self):
        shapes = [(2, 32, 24, 30), (2, 32, 33), (24, 24)]
        indices_shapes = [(2, 2), (2, 3), (1, 3)]
        input_dtypes = [torch.bfloat16]
        accumulate_list = [True, False]
        for items in product(shapes, input_dtypes, indices_shapes, accumulate_list):
            # items[0]: input shape
            # items[1]: input data type
            # items[2]: indices shape
            # items[3]: whether to accumulate the source to input
            input = torch.randn(items[0], dtype=torch.float)
            idx = torch.linspace(0, input.numel() - 1, np.prod(items[2]))
            idx = idx.to(torch.long).reshape(items[2])
            source = torch.rand_like(idx, dtype=torch.float)
            input = input.to(items[1])
            source = source.to(items[1])
            input_mlu = input.mlu()
            out_cpu = input.put_(idx, source, items[3])
            out_mlu = input_mlu.put_(idx.mlu(), source.mlu(), items[3])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
        # test bfloat16 backward
        input = torch.randn((2, 5, 3), dtype=torch.bfloat16)
        input_cpu = input.float()
        input_cpu.requires_grad = True
        input_mlu = self.to_device(input)
        input_mlu.requires_grad = True
        idx_cpu = torch.LongTensor([[0], [-2]])
        idx_mlu = self.to_device(idx_cpu)
        source = torch.LongTensor([[3], [4]]).to(dtype=torch.bfloat16)
        tensor_cpu = source.float()
        tensor_mlu = self.to_device(source)
        out_cpu = torch.Tensor.put(input_cpu, idx_cpu, tensor_cpu)
        out_mlu = torch.Tensor.put(input_mlu, idx_mlu, tensor_mlu)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = self.to_mlu_dtype(grad, torch.bfloat16)
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            input_cpu.grad, input_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("46GB")
    def test_put__large_bfloat16(self):
        shapes = [(5, 128, 1024, 1024)]
        indices_shapes = [(2, 2)]
        input_dtypes = [torch.bfloat16]
        accumulate_list = [True]
        for items in product(shapes, input_dtypes, indices_shapes, accumulate_list):
            # items[0]: input shape
            # items[1]: input data type
            # items[2]: indices shape
            # items[3]: whether to accumulate the source to input
            input = torch.randn(items[0], dtype=items[1])
            idx = torch.linspace(0, (input.numel() - 1) / 2, np.prod(items[2]))
            idx = idx.to(torch.long).reshape(items[2])
            source = torch.rand_like(idx, dtype=items[1])
            input_mlu = input.mlu()
            out_cpu = input.put_(idx, source, items[3])
            out_mlu = input_mlu.put_(idx.mlu(), source.mlu(), items[3])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
