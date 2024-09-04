from __future__ import print_function

import sys
import logging
import os
import copy
import unittest
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
    read_card_info,
    skipBFloat16IfNotSupport,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestIndexFillOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill(self):
        shape_list = [(0, 1, 2), (2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 0], [0, 2, 1], [0, 2, 1, 2, 1], [0, 1, 2], [0, 2, 2]]
        dim_list = [1, -2, 1, 2, 0]
        for i, shape in enumerate(shape_list):
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            index = torch.tensor(index_list[i])
            index_mlu = index.to("mlu")
            out_cpu = torch.index_fill(x, dim_list[i], index, 2)
            out_mlu = torch.index_fill(x_mlu, dim_list[i], index_mlu, 2)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_backward(self):
        x = torch.randn(4, 3, 2, 3, dtype=torch.float, requires_grad=True)
        x_mlu = x.to("mlu")
        index = torch.tensor([0, 2, 1, 2, 1])
        index_mlu = index.to("mlu")

        out_cpu = torch.index_fill(x, 1, index, 2)
        out_mlu = torch.index_fill(x_mlu, 1, index_mlu, 2)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

        grad = torch.randn(out_cpu.shape)
        grad_mlu = copy.deepcopy(grad).to("mlu")
        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_dtype(self):
        shape_list = [(2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3)]
        type_list = [
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.int32,
            torch.uint8,
            torch.long,
            torch.double,
            torch.bool,
        ]
        for t in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor([0, 2])
                index_mlu = index.to("mlu")
                x.index_fill_(1, index, 2)
                x_mlu.index_fill_(1, index_mlu, 2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_memory_format(self):
        shape_list = [(3, 4, 5, 2, 3)]
        dim_list = [-1, 0, 1, 2, 3, 4]
        memory_formats = [True, False]
        list_list = [shape_list, dim_list, memory_formats]
        for shape, dim, channel_last in product(*list_list):
            x = torch.randn(shape, dtype=torch.float)
            if channel_last is True:
                x = self.convert_to_channel_last(x)
            x_mlu = copy.deepcopy(x).to("mlu")
            index = torch.tensor([1])
            index_mlu = index.to("mlu")
            x.index_fill_(dim, index, -1)
            x_mlu.index_fill_(dim, index_mlu, -1)
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_not_dense(self):
        shape_list = [(2, 4, 5), (4, 4, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1], [0, 1, 2], [0, 2, 2]]
        dim_list = [-2, 1, 2, 0]
        for i, shape in enumerate(shape_list):
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            index = torch.tensor(index_list[i])
            index_mlu = index.to("mlu")
            out_cpu = torch.index_fill(x[:, :3, ...], dim_list[i], index, 1)
            out_mlu = torch.index_fill(x_mlu[:, :3, ...], dim_list[i], index_mlu, 1)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_inplace_not_dense(self):
        shape_list = [(2, 4, 5), (4, 4, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1], [0, 1, 2], [0, 2, 2]]
        dim_list = [-2, 1, 2, 0]
        for i, shape in enumerate(shape_list):
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            ori_ptr = x_mlu.data_ptr()
            index = torch.tensor(index_list[i])
            index_mlu = index.to("mlu")
            x[:, :3, ...].index_fill_(dim_list[i], index, 2)
            x_mlu[:, :3, ...].index_fill_(dim_list[i], index_mlu, 2)
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_value(self):
        shape_list = [(3, 4, 5, 2, 3)]
        for value in [-1, 0, 1, 1.2, -3.4]:
            for shape in shape_list:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor([1])
                index_mlu = index.to("mlu")
                x.index_fill_(1, index, value)
                x_mlu.index_fill_(1, index_mlu, value)
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_scalar_index(self):
        shape = (2, 3, 4)
        for value in [-1, 0, 1, 1.2, -3.4]:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            index = torch.tensor(2)
            index_mlu = index.to("mlu")
            x.index_fill_(1, index, value)
            x_mlu.index_fill_(1, index_mlu, value)
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_tensor_value(self):
        shape = (2, 3, 4)
        for value in [-1, 0, 1, 1.2, -3.4]:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            index = torch.tensor(2)
            index_mlu = index.to("mlu")
            value_tensor = torch.tensor(value)
            x.index_fill_(1, index, value_tensor)
            x_mlu.index_fill_(1, index_mlu, value_tensor.mlu())
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_fill_exception(self):
        shape = (2, 4, 5)
        x = torch.randn(shape, dtype=torch.float)
        x_mlu = copy.deepcopy(x).to("mlu")
        index = torch.randn(1, 2).to("mlu")

        ref_msg = "index_fill_(): Expected dtype int64 for index."
        with self.assertRaises(IndexError) as info:
            x_mlu.index_fill_(1, index, -1)
        self.assertEqual(info.exception.args[0], ref_msg)

        ref_msg = "index_fill_(): Expected dtype int64 for index."
        with self.assertRaises(IndexError) as info:
            x_mlu.index_fill_(1, index.bool(), -1)

        self.assertEqual(info.exception.args[0], ref_msg)
        device = "mlu"
        dtype = torch.half
        x = torch.tensor([[1, 2], [4, 5]], dtype=dtype, device=device)
        index = torch.tensor([0], device=device)
        with self.assertRaises(RuntimeError) as info:
            x.index_fill_(1, index, 1 + 1j)
        msg = "index_fill_(): Converting complex Scalar to non-complex type is not supported"
        self.assertEqual(info.exception.args[0], msg)
        with self.assertRaises(RuntimeError) as info:
            x.index_fill_(1, index, torch.tensor([1]))
        msg = (
            "index_fill_ only supports a 0-dimensional value tensor,"
            " but got tensor with 1 dimension(s)."
        )
        self.assertEqual(info.exception.args[0], msg)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_index_fill_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        type_list = [torch.int8]
        for t in type_list:
            for shape in shape_list:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor([0, 2])
                index_mlu = index.to("mlu")
                x.index_fill_(1, index, 2)
                x_mlu.index_fill_(1, index_mlu, 2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_index_fill_bfloat16(self):
        x = torch.randn(4, 3, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        x_mlu = x.to("mlu")
        index = torch.tensor([0, 2, 1, 2, 1])
        index_mlu = index.to("mlu")

        out_cpu = torch.index_fill(x, 1, index, 2)
        out_mlu = torch.index_fill(x_mlu, 1, index_mlu, 2)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        grad_mlu = copy.deepcopy(grad).to("mlu")
        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.0, use_MSE=True)


if __name__ == "__main__":
    run_tests()
