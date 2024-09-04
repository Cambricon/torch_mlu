from __future__ import print_function

import sys
import os
from itertools import product
import copy
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestEmbeddingOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_embedding(self):
        x_shape_lists = [
            (51, 77),
            (32,),
            (16, 7200, 2),
            (16, 7200, 1, 1),
            (2, 3, 4, 5, 6),
            (0,),
            (),
        ]
        c_lists = [9796, 1234]
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        func_list = [
            lambda x: x,
            self.convert_to_channel_last,
            self.get_not_contiguous_tensor,
        ]
        for c, x_shape, func in product(c_lists, x_shape_lists, func_list):
            for data_type, err in dtype_list:
                x_0 = torch.randint(0, c, x_shape)
                x = func(copy.deepcopy(x_0))
                matrix = torch.rand(c, 511, dtype=torch.float)
                out_cpu = torch.nn.functional.embedding(x, matrix)
                out_mlu = torch.nn.functional.embedding(
                    x.to("mlu"), self.to_mlu_dtype(matrix, data_type)
                )
                self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_backward(self):
        x_shape_lists = [
            (2, 4),
            (32,),
            (16, 7200, 2),
            (16, 7200, 1, 1),
            (2, 3, 4, 5, 6),
            (0,),
            (),
        ]
        c_lists = [8, 16]
        data_types = [torch.float, torch.half]
        values = [0.0001, 0.003]
        for c, x_shape in product(c_lists, x_shape_lists):
            for i in range(2):
                x = torch.randint(0, c, x_shape)
                matrix = torch.rand(c, 3, dtype=torch.float, requires_grad=True)
                out_cpu = torch.nn.functional.embedding(x, matrix)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                g1 = copy.deepcopy(matrix.grad)
                matrix.grad.zero_()
                out_mlu = torch.nn.functional.embedding(
                    self.to_device(x), self.to_mlu_dtype(matrix, data_types[i])
                )
                out_mlu.backward(self.to_mlu_dtype(grad, data_types[i]))
                g2 = copy.deepcopy(matrix.grad)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), values[i], use_MSE=True
                )
                self.assertTensorsEqual(g1, g2.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_exception(self):
        x = torch.randint(0, 9796, [51, 77])
        matrix = torch.tensor(1.0)
        ref_msg_1 = "'weight' must be 2-D"
        with self.assertRaisesRegex(RuntimeError, ref_msg_1):
            torch.nn.functional.embedding(self.to_mlu(x), self.to_device(matrix))

        x = torch.randint(0, 9796, [51, 77]).float()
        matrix = torch.rand(9796, 511, dtype=torch.float)
        ref_msg_3 = (
            "Expected tensor for argument #1 'indices' to have one of the following"
        )
        ref_msg_3 = ref_msg_3 + " scalar types: Long, Int; but got mluFloatType instead"
        ref_msg_3 = ref_msg_3 + r" \(while checking arguments for embedding\)"
        with self.assertRaisesRegex(RuntimeError, ref_msg_3):
            torch.nn.functional.embedding(self.to_device(x), self.to_mlu(matrix))

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_embedding_bfloat16(self):
        x = torch.randint(0, 9796, (20, 30))
        matrix = torch.rand(9796, 1024, dtype=torch.bfloat16).float()
        matrix_cpu = torch.nn.Parameter(matrix)
        matrix_mlu = torch.nn.Parameter(matrix.mlu().bfloat16())
        out_cpu = torch.nn.functional.embedding(x, matrix_cpu)
        out_mlu = torch.nn.functional.embedding(x.mlu(), matrix_mlu)
        grad = torch.randn_like(out_cpu).bfloat16()
        out_cpu.backward(grad.float())
        out_mlu.backward(grad.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            matrix_cpu.grad.float(), matrix_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_with_padding_idx(self):
        x = torch.randint(0, 123, (32, 54), dtype=torch.int)
        x_mlu = x.to("mlu")
        weight = torch.randn(size=(123, 10), dtype=torch.float)
        weight_mlu = weight.to("mlu")
        out_mlu = torch.nn.functional.embedding(x_mlu, weight_mlu, padding_idx=14)
        out_cpu = torch.nn.functional.embedding(x, weight, padding_idx=14)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_transpose_backward(self):
        x_shape = (2, 6)
        c = 8
        x = torch.randint(0, c, x_shape)
        matrix = torch.rand(c, 3, requires_grad=True)
        out_cpu = torch.nn.functional.embedding(x, matrix).transpose(0, 1).contiguous()
        grad = torch.randn(out_cpu.shape, dtype=torch.float)
        out_cpu.backward(torch.ones_like(out_cpu))
        g1 = copy.deepcopy(matrix.grad)
        matrix.grad.zero_()

        out_mlu = (
            torch.nn.functional.embedding(self.to_device(x), self.to_device(matrix))
            .transpose(0, 1)
            .contiguous()
        )
        out_mlu.backward(torch.ones_like(out_mlu))
        g2 = copy.deepcopy(matrix.grad)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.0001, use_MSE=True)
        self.assertTensorsEqual(g1, g2.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_embedding_backward_with_mix_memory_format(self):
        grad_output = self.convert_to_channel_last(torch.rand((1, 5, 5, 15)))
        indices = torch.randint(0, 5, size=(1, 5, 5), dtype=torch.int)
        model = torch.ops.aten.embedding_dense_backward
        out_cpu = model(grad_output, indices, 16, -1, False)
        out_mlu = model(
            self.to_device(grad_output), self.to_device(indices), 16, -1, False
        )
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("32GB")
    @testinfo()
    def test_embedding_backward_large(self):
        data_types = [torch.float, torch.half]
        values = [0.0001, 0.003]
        c = 16
        shape = [1, 1024, 1024, 512]
        x = torch.randint(0, c, shape)
        matrix = torch.rand(c, 3, dtype=torch.float, requires_grad=True)

        out_cpu = torch.nn.functional.embedding(x, matrix)
        grad = torch.randn(out_cpu.shape, dtype=torch.float)
        out_cpu.backward(grad)
        g1 = copy.deepcopy(matrix.grad)

        matrix.grad.zero_()
        out_mlu = torch.nn.functional.embedding(self.to_device(x), self.to_mlu(matrix))
        out_mlu.backward(self.to_mlu(grad))
        g2 = copy.deepcopy(matrix.grad)

        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
        self.assertTensorsEqual(g1, g2.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
