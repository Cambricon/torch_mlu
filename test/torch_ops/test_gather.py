from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (  # pylint: disable=C0413, C0411
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)

logging.basicConfig(level=logging.DEBUG)


class TestGatherOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_gather(self):
        shapes = [
            (32, 12, 24, 32, 16),
            (32, 3, 224, 224),
            (2, 100, 56),
            (234, 32),
            (0, 32),
            (24,),
        ]
        dtypes = [torch.double, torch.float, torch.long, torch.int]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                for t in dtypes:
                    x = torch.randn(shape, dtype=torch.float).to(t)
                    index = torch.abs(
                        torch.rand(shape, dtype=torch.float) * shape[dim]
                    ).to(torch.int64)
                    out = torch.gather(x, dim, index)
                    x_mlu = self.to_mlu(x)
                    index_mlu = self.to_device(index)
                    out_mlu = torch.gather(x_mlu, dim, index_mlu)
                    self.assertTensorsEqual(
                        out, out_mlu.cpu().float(), 0.000, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_out(self):
        shape_list = [
            (32, 12, 24, 32, 16),
            (32, 3, 224, 224),
            (2, 100, 56),
            (234, 32),
            (0, 32),
            (24,),
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, func in product(shape_list, func_list):
            for dim in range(-len(shape), len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                index = torch.abs(torch.rand(shape, dtype=torch.float) * shape[dim]).to(
                    torch.int64
                )
                out_cpu = torch.randn(32, 3, 224, 224)
                out_cpu = func(out_cpu)
                torch.gather(x, dim, index, out=out_cpu)
                x_mlu = x.to("mlu")
                index_mlu = index.to("mlu")
                out_mlu = torch.randn(32, 3, 224, 224).to("mlu")
                out_mlu = func(out_mlu)
                torch.gather(x_mlu, dim, index_mlu, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.000, use_MSE=True
                )
                self.assertTrue(out_cpu.stride() == out_mlu.stride())
                self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_channelslast_and_nodense(self):
        def run_test(x, x_mlu, dim, index):
            out_cpu = torch.gather(x, dim, index)
            out_mlu = torch.gather(x_mlu, dim, index.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.000, use_MSE=True)

        shapes = [(32, 3, 224, 224), (2, 100, 56, 56), (234, 3, 32, 32)]
        dims = [0, 1, 2, 3]
        for shape in shapes:
            for dim in dims:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = x.to("mlu")
                index = torch.abs(torch.rand(shape, dtype=torch.float) * shape[dim]).to(
                    torch.int64
                )
                # channels_last input
                run_test(
                    x.to(memory_format=torch.channels_last),
                    x_mlu.to(memory_format=torch.channels_last),
                    dim,
                    index,
                )

                # not-dense input
                shape = x[..., :2].shape
                index = torch.randint(0, shape[dim], shape)
                run_test(x[..., :2], x_mlu[..., :2], dim, index)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_with_stride_and_high_dims(self):
        expand_shapes = [
            (8192, 4096),
            (543, 256, 32),
            (10, 4, 5, 5, 6, 7),
            (1372, 4, 32),
            (1024, 512, 32, 16),
            (8, 8, 4, 8, 4, 8, 2, 8),
        ]
        org_shapes = [
            (8192, 1),
            (1, 256, 1),
            (10, 4, 5, 1, 1, 1),
            (1372, 1, 1),
            (1024, 512, 1, 1),
            (8, 1, 4, 1, 4, 1, 2, 1),
        ]
        for expand_shape, org_shape in zip(expand_shapes, org_shapes):
            idx = torch.randint(0, 2, org_shape)
            idx_mlu = idx.mlu()
            src = torch.randn(expand_shape)
            src_mlu = src.mlu()
            for dim in range(len(expand_shape)):
                cpu_out = torch.gather(src, dim, idx.expand(expand_shape))
                mlu_out = torch.gather(src_mlu, dim, idx_mlu.expand(expand_shape))
                self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_exception(self):
        shape = (2, 100, 56)
        dim = 0
        x = torch.randn(shape, dtype=torch.float)
        index = torch.randint(0, shape[dim], shape)
        x_mlu = self.to_mlu(x)
        index_mlu = self.to_device(index)
        ref_msg = r"gather\(\): Expected dtype int64 for index"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(x_mlu, dim, index_mlu.float())
        index_mlu = self.to_device(index)
        ref_msg = "Index tensor must have the same number of dimensions as input tensor"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(x_mlu, dim, index_mlu.resize_(100, 112))
        index_mlu = self.to_device(index)
        src = torch.rand((6,), device="mlu")
        index = torch.tensor([2, 1, 0], device="mlu", dtype=torch.int64)
        ref_msg = r"unsupported operation: some elements of the input tensor and "
        ref_msg += r"the written-to tensor refer to a single memory location\. "
        ref_msg += r"Please clone\(\) the tensor before performing the operation."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(src, 0, index, out=src)
        index = torch.tensor([2, 1, 0], device="mlu", dtype=torch.int64)
        ref_msg = r"unsupported operation: some elements of the input tensor and "
        ref_msg += r"the written-to tensor refer to a single memory location\. "
        ref_msg += r"Please clone\(\) the tensor before performing the operation."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.gather(index.clone(), 0, index[1:], out=index[:1])

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_zero(self):
        s = torch.ones(size=[], dtype=torch.float)
        index = torch.zeros(size=(1,), dtype=torch.long)
        a = torch.gather(s, 0, index)
        b = torch.gather(s.to("mlu"), 0, index.to("mlu"))
        self.assertTensorsEqual(a, b.cpu(), 0.000, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_index_zero(self):
        input = torch.randn((1,))
        out_cpu = torch.gather(input, 0, torch.zeros((), dtype=torch.int64))
        out_mlu = torch.gather(input.mlu(), 0, torch.zeros((), dtype=torch.int64).mlu())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.000, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_dtype(self):
        shape = (2, 100, 56)
        dim = 0
        dtype_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.short,
            torch.bool,
        ]
        for data_dtype in dtype_list:
            input = 100 * torch.rand(shape)
            input = input.to(data_dtype)
            index = torch.abs(torch.rand(shape, dtype=torch.float) * shape[dim]).to(
                torch.int64
            )
            out = torch.gather(input, dim, index)
            input_mlu = self.to_mlu(input)
            index_mlu = self.to_device(index)
            out_mlu = torch.gather(input_mlu, dim, index_mlu)
            self.assertTensorsEqual(
                out.double(), out_mlu.cpu().double(), 0.000, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_gather_bfloat16(self):
        shape = (2, 100, 56)
        dim = 0
        dtype_list = [torch.bfloat16]
        for data_dtype in dtype_list:
            input = 100 * torch.rand(shape)
            input = input.to(data_dtype)
            index = torch.abs(torch.rand(shape, dtype=torch.float) * shape[dim]).to(
                torch.int64
            )
            out = torch.gather(input, dim, index)
            input_mlu = self.to_mlu(input)
            index_mlu = self.to_device(index)
            out_mlu = torch.gather(input_mlu, dim, index_mlu)
            self.assertTensorsEqual(
                out.double(), out_mlu.cpu().double(), 0.000, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_zero_numel(self):
        shape = [(0, 1, 2, 0), (0, 1, 3, 0)]
        x = torch.randn(shape[0])
        larger_shape = torch.empty(shape[1], dtype=torch.int64)
        out_cpu = torch.gather(x, 2, larger_shape)
        out_mlu = torch.gather(x.mlu(), 2, larger_shape.mlu())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("66GB")
    def test_gather_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        dtype_list = [
            torch.half,
        ]
        list_list = [shape_list, dtype_list]
        dim = 3
        for shape, dtype in product(*list_list):
            x = torch.randn(shape, dtype=torch.float)
            index = torch.abs(torch.rand(shape, dtype=torch.float) * shape[dim]).to(
                torch.int64
            )
            out = torch.gather(x, dim, index)
            x_mlu = self.to_mlu_dtype(x, dtype)
            index_mlu = self.to_device(index)
            out_mlu = torch.gather(x_mlu, dim, index_mlu)
            self.assertTensorsEqual(
                out.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_dim(self):
        N = 5
        T = 6
        S = 7
        d = 8
        input = torch.randn(N, T, S, d)
        index = torch.randint(0, 5, (N, T, S - 1, 2))
        out_cpu = torch.gather(input=input, dim=3, index=index)
        input = input.to("mlu")
        index = index.to("mlu")
        out_mlu = torch.gather(input=input, dim=3, index=index)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_gather_overlap(self):
        src = torch.ones((2, 5), device="mlu", requires_grad=True)
        index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]], device="mlu")
        x = torch.zeros(3, 5, dtype=src.dtype, device="mlu")
        y = x.scatter_add_(0, index, src).sum()
        y.backward()


if __name__ == "__main__":
    run_tests()
