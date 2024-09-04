from __future__ import print_function

import sys
import os
import unittest
import logging

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
)

logging.basicConfig(level=logging.DEBUG)


class TestScatterOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_scatter(self):
        device = "mlu"
        shape = (0, 1, 2, 0)
        # scatter
        for dim in [0, 2]:
            y = torch.randn(shape, device=device)
            y_src = torch.randn(shape, device=device)
            ind = torch.empty(shape, dtype=torch.int64, device=device)
            self.assertEqual(shape, y.scatter_(dim, ind, y_src).shape)

        z = torch.randn((2, 3, 4), device=device)
        z_src = torch.randn((2, 3, 4), device=device)
        self.assertEqual(
            z,
            z.scatter_(
                2, torch.empty((2, 3, 0), dtype=torch.int64, device=device), z_src
            ),
        )

        # test index[d] <= src[d]
        input = torch.zeros(4, 4, device=device)
        src = torch.ones(2, 2, device=device)
        index = torch.tensor([[1], [2]], device=device, dtype=torch.long)
        input.scatter_(0, index, src)
        self.assertEqual(
            input,
            torch.tensor(
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                device=device,
                dtype=torch.float32,
            ),
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_dtype(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2)]
        supported_dtype = [
            torch.float64,
            torch.float32,
            torch.half,
            torch.long,
            torch.int32,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.bool,
        ]
        float_dtype = [torch.float64, torch.float32, torch.half]
        for dt in supported_dtype:
            for shape in shapes:
                dim = 1
                index = torch.randint(0, shape[dim], shape)
                input_cpu = torch.randn(shape)
                if dt not in float_dtype:
                    input_cpu = torch.randint(-10, 10, shape)
                input_cpu = input_cpu.to(dt)
                input_mlu = input_cpu.to("mlu")
                out_cpu = torch.scatter(input_cpu, dim, index, 1)
                out_mlu = torch.scatter(input_mlu, dim, index.to("mlu"), 1)
                self.assertEqual(out_mlu, out_cpu)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_scatter_bfloat16(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2)]
        supported_dtype = [torch.bfloat16]
        for dt in supported_dtype:
            for shape in shapes:
                dim = 1
                index = torch.randint(0, shape[dim], shape)
                input_cpu = torch.randn(shape)
                input_cpu = input_cpu.to(dt)
                input_mlu = input_cpu.to("mlu")
                out_cpu = torch.scatter(input_cpu, dim, index, 1)
                out_mlu = torch.scatter(input_mlu, dim, index.to("mlu"), 1)
                self.assertEqual(out_mlu, out_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_channels_last(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2)]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                index = torch.randint(0, shape[dim], shape)
                input_cpu = torch.randn(shape)
                out_cpu = torch.scatter(input_cpu, dim, index, 1)
                input_mlu = self.convert_to_channel_last(input_cpu).to("mlu")
                out_mlu = torch.scatter(input_mlu, dim, index.to("mlu"), 1)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

            for dim in range(-len(shape), len(shape)):
                index = torch.randint(0, shape[dim], shape)
                input_cpu = torch.randn(shape)
                out_cpu = torch.scatter(input_cpu, dim, index, 1)
                input_mlu = self.convert_to_channel_last(input_cpu).to("mlu")
                index_mlu = self.convert_to_channel_last(index).to("mlu")
                out_mlu = torch.scatter(input_mlu, dim, index_mlu, 1)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_zero_element(self):
        a = torch.randn(2, 3, 4)
        a_mlu = a.to("mlu")
        index = torch.randn(
            0,
        ).to(dtype=torch.long)
        index_mlu = index.to("mlu")
        a.scatter_(2, index, 0)
        a_mlu.scatter_(2, index_mlu, 0)
        self.assertTensorsEqual(a, a_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_with_stride(self):
        input_shape = (100, 512, 2, 5)
        index_shape = (100, 512, 2, 1)
        dim = 2
        index = torch.randint(0, index_shape[dim], index_shape)
        input_cpu = torch.randn(input_shape)
        out_cpu = torch.scatter(input_cpu, dim, index.expand(100, 512, 2, 5), 1)
        input_mlu = input_cpu.mlu()
        out_mlu = torch.scatter(input_mlu, dim, index.mlu().expand(100, 512, 2, 5), 1)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_with_stride_and_high_dims(self):
        expand_shapes = [
            (543, 256, 32),
            (10, 4, 5, 5, 6, 7),
            (1372, 4, 32),
            (1024, 512, 32, 16),
            (8, 8, 4, 8, 4, 8, 2, 8),
        ]
        org_shapes = [
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
                cpu_out = torch.scatter(src, dim, idx.expand(expand_shape), 1)
                mlu_out = torch.scatter(src_mlu, dim, idx_mlu.expand(expand_shape), 1)
                self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2)]
        for shape in shapes:
            dim = 1
            index = torch.randint(0, shape[dim], shape)
            input_cpu = torch.randn(shape)
            input_mlu = input_cpu.mlu()
            out_cpu = torch.scatter(input_cpu, dim, index, 1, reduce="add")
            out_mlu = torch.scatter(input_mlu, dim, index.mlu(), 1, reduce="add")
            self.assertEqual(out_mlu, out_cpu)

            src = torch.randn(shape)
            out_cpu = torch.scatter(input_cpu, dim, index, src, reduce="add")
            out_mlu = torch.scatter(
                input_mlu, dim, index.mlu(), src.mlu(), reduce="add"
            )
            self.assertEqual(out_mlu, out_cpu)

            out_cpu = torch.scatter_reduce(input_cpu, dim, index, src, reduce="sum")
            out_mlu = torch.scatter_reduce(
                input_mlu, dim, index.mlu(), src.mlu(), reduce="sum"
            )
            self.assertEqual(out_mlu, out_cpu)

            out_cpu = torch.scatter_reduce(
                input_cpu, dim, index, src, reduce="sum", include_self=False
            )
            out_mlu = torch.scatter_reduce(
                input_mlu, dim, index.mlu(), src.mlu(), reduce="sum", include_self=False
            )
            self.assertEqual(out_mlu, out_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_exception(self):
        a = torch.randn((4, 4), dtype=torch.complex64).mlu()
        index = torch.randint(0, 3, size=(2, 2), dtype=torch.long).mlu()
        src = torch.randn(size=(2, 2), dtype=torch.complex64).mlu()
        ref_msg = f'"MLU scatter" not implemented for'
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.scatter_(0, index, src)

        a = torch.randn((4, 4)).mlu()
        index = torch.randint(0, 3, size=(2, 2), dtype=torch.long).mlu()
        src = torch.randn(size=(2, 2)).mlu()
        ref_msg = f"MLU scatter reduce of prod is not supported"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.scatter_(0, index, src, reduce="multiply")

        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.scatter_(0, index, 1, reduce="multiply")

        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.scatter_reduce_(0, index, src, reduce="prod")

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("68GB")
    def test_scatter_large(self):
        shapes = [(1024, 1024, 1024, 5)]
        dtype = torch.half
        for shape in shapes:
            dim = 1
            index = torch.randint(0, shape[dim], shape)
            input_cpu = torch.randn(shape)
            input_cpu = input_cpu.to(dtype)
            input_mlu = input_cpu.to("mlu")
            out_cpu = torch.scatter(input_cpu, dim, index, 1)
            out_mlu = torch.scatter(input_mlu, dim, index.to("mlu"), 1)
            self.assertEqual(out_mlu, out_cpu)


if __name__ == "__main__":
    run_tests()
