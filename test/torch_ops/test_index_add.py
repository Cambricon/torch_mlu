from __future__ import print_function

import sys
import logging
import os
import copy
import unittest
import torch

import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


def get_memory_format(dim):
    if dim == 4:
        return torch.channels_last
    elif dim == 5:
        return torch.channels_last_3d
    else:
        return torch.contiguous_format


class TestIndexAddOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_index_add(self):
        # FIXME(guyi):CTR-3882, now cnnl kernel doesn't support self and source expand operation  # pylint: disable=W0511
        # so must keep these two have same shape except dim-th dimension
        shape_list = [(0, 1, 2), (2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 0], [0, 2, 1], [0, 2, 1, 2, 1], [0, 1, 2], [0, 2, 2]]
        source_list = [(0, 2, 2), (2, 3, 5), (4, 5, 2, 3), (3, 4, 3, 2, 3), (3, 4)]
        dim_list = [1, -2, 1, 2, 0]
        channels_last_list = [True, False]
        for i, shape in enumerate(shape_list):
            for channels_last in channels_last_list:
                memory_type = torch.contiguous_format
                if channels_last is True:
                    memory_type = get_memory_format(len(shape))
                x = torch.randn(shape, dtype=torch.float).to(memory_format=memory_type)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor(index_list[i])
                index_mlu = index.to("mlu")
                source = torch.rand(source_list[i])
                out_cpu = torch.index_add(x, dim_list[i], index, source)
                out_mlu = torch.index_add(
                    x_mlu, dim_list[i], index_mlu, source.to("mlu")
                )
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_not_dense(self):
        shape_list = [(2, 4, 5), (4, 4, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1], [0, 1, 2], [0, 2, 2]]
        source_list = [(2, 5, 5), (4, 5, 2, 3), (3, 4, 3, 2, 3), (3, 4)]
        dim_list = [-2, 1, 2, 0]
        dtypes = [torch.double, torch.float, torch.long, torch.int]
        for i, shape in enumerate(shape_list):
            for t in dtypes:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor(index_list[i])
                index_mlu = index.to("mlu")
                source = torch.rand(source_list[i]).to(t)
                out_cpu = torch.index_add(
                    x[:, :3, ...], dim_list[i], index, source[:, :3, ...]
                )
                out_mlu = torch.index_add(
                    x_mlu[:, :3, ...],
                    dim_list[i],
                    index_mlu,
                    source.to("mlu")[:, :3, ...],
                )
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_inplace_not_dense(self):
        shape_list = [(2, 4, 5), (4, 4, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1], [0, 1, 2], [0, 2, 2]]
        source_list = [(2, 5, 5), (4, 5, 2, 3), (3, 4, 3, 2, 3), (3, 4)]
        dim_list = [-2, 1, 2, 0]
        for i, shape in enumerate(shape_list):
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            ori_ptr = x_mlu.data_ptr()
            index = torch.tensor(index_list[i])
            index_mlu = index.to("mlu")
            source = torch.rand(source_list[i])
            x[:, :3, ...].index_add(dim_list[i], index, source[:, :3, ...])
            x_mlu[:, :3, ...].index_add(
                dim_list[i], index_mlu, source.to("mlu")[:, :3, ...]
            )
            self.assertEqual(ori_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_inplace(self):
        shape_list = [(2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1, 2, 1], [0, 1, 2], [0, 2, 2]]
        source_list = [(2, 3, 5), (4, 5, 2, 3), (3, 4, 3, 2, 3), (3, 4)]
        dim_list = [-2, 1, 2, 0]
        channels_last_list = [True, False]
        for i, shape in enumerate(shape_list):
            for channels_last in channels_last_list:
                memory_type = torch.contiguous_format
                if channels_last is True:
                    memory_type = get_memory_format(len(shape))
                x = torch.randn(shape, dtype=torch.float).to(memory_format=memory_type)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor(index_list[i])
                index_mlu = index.to("mlu")
                source = torch.rand(source_list[i])
                x.index_add_(dim_list[i], index, source)
                ori_ptr = x_mlu.data_ptr()
                x_mlu.index_add_(dim_list[i], index_mlu, source.to("mlu"))
                self.assertEqual(ori_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_dtype(self):
        shape_list = [(2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1, 2, 1], [0, 1, 2], [0, 2, 2]]
        source_list = [(2, 3, 5), (4, 5, 2, 3), (3, 4, 3, 2, 3), (3, 4)]
        dim_list = [-2, 1, 2, 0]
        type_list = [
            torch.float,
            torch.short,
            torch.int32,
            torch.int16,
            torch.double,
            torch.long,
        ]
        for i, shape in enumerate(shape_list):
            for t in type_list:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor(index_list[i])
                index_mlu = index.to("mlu")
                source = torch.rand(source_list[i]).to(t)
                x.index_add_(dim_list[i], index, source)
                ori_ptr = x_mlu.data_ptr()
                x_mlu.index_add_(dim_list[i], index_mlu, source.to("mlu"))
                self.assertEqual(ori_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_backward(self):
        x = torch.zeros((2, 4, 5), dtype=torch.float, device="cpu", requires_grad=True)
        x_mlu = torch.zeros(
            (2, 4, 5), dtype=torch.float, device="mlu", requires_grad=True
        )
        index = torch.tensor([0, 2, 1])
        index_mlu = torch.tensor([0, 2, 1], device="mlu")
        dim = -2
        source = torch.rand((2, 3, 5), dtype=torch.float)
        source_mlu = source.mlu()
        out_cpu = torch.index_add(x, dim, index, source)
        out_mlu = torch.index_add(x_mlu, dim, index_mlu, source_mlu)
        grad = torch.randn(2, 4, 5)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        out_grad = x.grad
        out_grad_mlu = x_mlu.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_add_exception(self):
        shape_list = [(2, 4, 5)]
        index_list = [[0, 2, 1]]
        source_list = [(2, 3, 5)]
        dim_list = [-2]
        x = torch.randn(shape_list[0], dtype=torch.float).to("mlu")
        index = torch.tensor(index_list[0]).to("mlu")
        source = torch.rand(source_list[0]).to("mlu")
        dim = dim_list[0]

        ref_msg = r"\"index_add\" not implemented for"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.index_add(x.to(torch.bool), dim, index, source.to(torch.bool))
        ref_msg = r"index_add_\(\): Index is supposed to be a vector"
        with self.assertRaisesRegex(IndexError, ref_msg):
            index = torch.tensor([[0, 2, 1]]).to("mlu")
            torch.index_add(x, dim, index, source)
        index = torch.tensor(index_list[0]).to("mlu").to(torch.float)
        ref_msg = r"index_add_\(\): Expected dtype int32/int64 for index"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.index_add(x, dim, index, source)
        index = torch.tensor(index_list[0]).to("mlu")
        source = source.to(torch.int64)
        ref_msg = r"index_add_\(\): self \("
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.index_add(x, dim, index, source)
        ref_msg = r"index_add_\(\): Number of indices \("
        source = torch.rand((2, 4, 5)).to("mlu")
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.index_add(x, dim, index, source)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_index_add_bfloat16(self):
        shape_list = [(2, 4, 5), (4, 3, 2, 3), (3, 4, 5, 2, 3), (5, 4)]
        index_list = [[0, 2, 1], [0, 2, 1, 2, 1], [0, 1, 2], [0, 2, 2]]
        source_list = [(2, 3, 5), (4, 5, 2, 3), (3, 4, 3, 2, 3), (3, 4)]
        dim_list = [-2, 1, 2, 0]
        type_list = [torch.bfloat16]
        for i, shape in enumerate(shape_list):
            for t in type_list:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor(index_list[i])
                index_mlu = index.to("mlu")
                source = torch.rand(source_list[i]).to(t)
                x.index_add_(dim_list[i], index, source)
                ori_ptr = x_mlu.data_ptr()
                x_mlu.index_add_(dim_list[i], index_mlu, source.to("mlu"))
                self.assertEqual(ori_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("21GB")
    def test_index_add_large_float(self):
        shape_list = [(4 * 1025, 1024 * 1024)]
        index_list = [[0, 2, 1, 3]]
        source_list = [(4, 1024 * 1024)]
        dim_list = [0]
        type_list = [torch.float]
        for i, shape in enumerate(shape_list):
            for t in type_list:
                x = torch.randn(shape, dtype=torch.float).to(t)
                x_mlu = copy.deepcopy(x).to("mlu")
                index = torch.tensor(index_list[i])
                index_mlu = index.to("mlu")
                source = torch.rand(source_list[i]).to(t)
                x.index_add_(dim_list[i], index, source)
                ori_ptr = x_mlu.data_ptr()
                x_mlu.index_add_(dim_list[i], index_mlu, source.to("mlu"))
                self.assertEqual(ori_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
