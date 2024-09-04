from __future__ import print_function

import sys
import os
from itertools import product
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)

logging.basicConfig(level=logging.DEBUG)


class TestIndexSelectOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_index_select(self):
        shape_list = [
            (0, 4),
            (2, 4, 5),
            (8, 9, 10),
            (8, 9, 10, 11),
            (3, 4, 5, 6, 7),
            (8, 9, 10, 11, 12, 14),
            (8, 9, 10, 11, 12, 13, 14),
            (99, 30, 40),
        ]
        c_lists = [9796, 10, 8767]
        index_shape_list = [(3,), (0,), ()]
        type_list = [
            torch.half,
            torch.float,
            torch.uint8,
            torch.int8,
            torch.long,
            torch.double,
            torch.int,
            torch.short,
            torch.bool,
        ]
        for shape, t, index_shape in product(shape_list, type_list, index_shape_list):
            x = torch.testing.make_tensor(shape, dtype=t, device="cpu")
            index = torch.randint(0, 3, (index_shape)).long()
            if t.is_floating_point:
                x_cpu = torch.nn.Parameter(x)
                x_mlu = torch.nn.Parameter(x.mlu())
            else:
                x_cpu = x
                x_mlu = x.mlu()
            out_cpu = torch.index_select(x_cpu, 1, index)
            out_mlu = torch.index_select(x_mlu, 1, index.mlu())
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0)
            if t.is_floating_point:
                grad_cpu = torch.randn(out_cpu.size())
                grad_mlu = grad_cpu.to("mlu")
                out_cpu.backward(grad_cpu)
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_cpu.grad.float(), x_mlu.grad.cpu().float(), 0.03
                )

        # size use in transformer
        for c in c_lists:
            x = torch.rand(c, 512, dtype=torch.float)
            index = torch.randint(0, c, [320], dtype=torch.int)
            out_cpu = torch.index_select(x, 0, index.long())
            out_mlu = torch.index_select(x.mlu(), 0, index.long().mlu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_not_dense(self):
        shape_list = [(8, 9, 10, 20), (3, 10, 5, 6, 40)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            index = torch.tensor([1, 3, 2, 1, 2, 1, 4, 3, 5])
            x_mlu = x.mlu()[:, :, :, 10:16]
            index_mlu = index.mlu()[2:5]
            out_cpu = torch.index_select(x[:, :, :, 10:16], 1, index[2:5])
            out_mlu = torch.index_select(x_mlu, 1, index_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_channel_last(self):
        shape_list = [(8, 9, 10, 20), (3, 4, 5, 6, 40)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.mlu()
            index = torch.tensor([1, 3, 2])
            out_cpu = torch.index_select(self.convert_to_channel_last(x), 1, index)
            out_mlu = torch.index_select(
                self.convert_to_channel_last(x_mlu), 1, index.mlu()
            )
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_with_mlu_scalar(self):
        x = torch.randn((), dtype=torch.float)
        x_mlu = x.mlu()
        index = torch.tensor(0)
        out_cpu = torch.index_select(x, 0, index)
        out_mlu = torch.index_select(x_mlu, 0, index.mlu())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)
        out_cpu = torch.randn((), dtype=torch.float)
        out_mlu = out_cpu.to("mlu")
        torch.index_select(x, 0, index, out=out_cpu)
        torch.index_select(x_mlu, 0, index.mlu(), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_with_nan_inf(self):
        x = torch.randn((2, 3), dtype=torch.float)
        x[0][0] = float("inf")
        x[0][1] = float("nan")
        x[0][2] = float("-inf")
        x_mlu = x.mlu()
        index = torch.tensor([0])
        out_cpu = torch.index_select(x, 0, index)
        out_mlu = torch.index_select(x_mlu, 0, index.mlu())
        self.assertEqual(out_cpu, out_mlu.cpu())

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("38GB")
    def test_index_select_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        c_lists = [9796, 10, 8767]
        index_shape_list = [(3,), (0,), ()]
        type_list = [torch.half, torch.bool]
        for shape, t, index_shape in product(shape_list, type_list, index_shape_list):
            x = torch.randn(shape, dtype=torch.float).to(t)
            index = torch.randint(0, 3, (index_shape))
            for dim in [1]:
                out_cpu = torch.index_select(x, dim, index)
                out_mlu = torch.index_select(x.mlu(), dim, index.long().mlu())
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0)
        # size use in transformer
        for c in c_lists:
            for dim in [0]:
                x = torch.rand(c, 512, dtype=torch.float)
                index = torch.randint(0, c, [320], dtype=torch.int)
                out_cpu = torch.index_select(x, dim, index.long())
                out_mlu = torch.index_select(x.mlu(), dim, index.long().mlu())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_index_select_bfloat16(self):
        shape_list = [
            (0, 4),
            (2, 4, 5),
            (8, 9, 10),
            (8, 9, 10, 11),
            (3, 4, 5, 6, 7),
            (8, 9, 10, 11, 12, 14),
            (8, 9, 10, 11, 12, 13, 14),
            (99, 30, 40),
        ]
        index_shape_list = [(3,), (0,), ()]
        type_list = [torch.bfloat16]
        for shape, t, index_shape in product(shape_list, type_list, index_shape_list):
            x = torch.testing.make_tensor(shape, dtype=t, device="cpu")
            index = torch.randint(0, 3, (index_shape)).long()
            if t.is_floating_point:
                x_cpu = torch.nn.Parameter(x)
                x_mlu = torch.nn.Parameter(x.mlu())
            else:
                x_cpu = x
                x_mlu = x.mlu()
            out_cpu = torch.index_select(x_cpu, 1, index)
            out_mlu = torch.index_select(x_mlu, 1, index.mlu())
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0)
            # index_select backward op is index_add, which bfloat16 is not supported yet.
            # if t.is_floating_point:
            #    grad_cpu = torch.randn(out_cpu.size())
            #    grad_mlu = grad_cpu.to('mlu')
            #    out_cpu.backward(grad_cpu)
            #    out_mlu.backward(grad_mlu)
            #    self.assertTensorsEqual(x_cpu.grad.float(), x_mlu.grad.cpu().float(), 0.03)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("38GB")
    def test_index_select_large_bfloat16(self):
        shape_list = [(5, 1024, 1024, 1024)]
        index_shape_list = [(3,), (0,), ()]
        type_list = [torch.bfloat16]
        for shape, t, index_shape in product(shape_list, type_list, index_shape_list):
            x = torch.randn(shape, dtype=torch.float).to(t)
            index = torch.randint(0, 3, (index_shape))
            for dim in [1]:
                out_cpu = torch.index_select(x, dim, index)
                out_mlu = torch.index_select(x.mlu(), dim, index.long().mlu())
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0)


class TestIndexSelectOutOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_out(self):
        shape_list = [(2, 4, 5), (8, 9, 10, 11), (8, 9, 10, 11, 12, 14)]
        c_lists = [9796, 10, 8767]
        type_list = [
            torch.half,
            torch.float,
            torch.long,
            torch.double,
            torch.int,
            torch.short,
            torch.bool,
            torch.uint8,
            torch.int8,
        ]
        for shape in shape_list:
            for t in type_list:
                x = torch.testing.make_tensor(shape, dtype=t, device="cpu")
                out_cpu = torch.testing.make_tensor(1, dtype=t, device="cpu")
                out_mlu = torch.testing.make_tensor(1, dtype=t, device="cpu").mlu()
                index = torch.tensor([1, 3, 2])
                torch.index_select(x, 1, index, out=out_cpu)
                torch.index_select(x.mlu(), 1, index.mlu(), out=out_mlu)
                self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0)
        # size use in transformer
        for c in c_lists:
            x = torch.rand(c, 512, dtype=torch.float)
            index = torch.randint(0, c, [320], dtype=torch.int)
            out_cpu = torch.index_select(x, 0, index.long())
            out_mlu = torch.index_select(x.mlu(), 0, index.long().mlu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_out_not_dense(self):
        shape_list = [(8, 9, 10, 20), (3, 10, 5, 6, 40)]
        for shape in shape_list:
            x = torch.testing.make_tensor(shape, dtype=torch.float, device="cpu")
            x_mlu = x.mlu()
            out_cpu = torch.testing.make_tensor(shape, dtype=torch.float, device="cpu")
            out_mlu = out_cpu.mlu()
            index = torch.tensor([1, 3, 2, 1, 2, 1, 4, 3, 5])
            index_mlu = index.mlu()
            torch.index_select(
                x[:, :, :, 10:16], 1, index[2:5], out=out_cpu[:, :, :, 10:16]
            )
            torch.index_select(
                x_mlu[:, :, :, 10:16], 1, index_mlu[2:5], out=out_mlu[:, :, :, 10:16]
            )
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_out_channel_last(self):
        shape_list = [(8, 9, 10, 20), (3, 4, 5, 6, 40)]
        out_cpu = torch.randn(1, dtype=torch.float)
        out_mlu = torch.randn(1, dtype=torch.float).to("mlu")
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.mlu()
            index = torch.tensor([1, 3, 2])
            index_mlu = index.mlu()
            torch.index_select(
                self.convert_to_channel_last(x),
                1,
                self.convert_to_channel_last(index),
                out=self.convert_to_channel_last(out_cpu),
            )
            torch.index_select(
                self.convert_to_channel_last(x_mlu),
                1,
                self.convert_to_channel_last(index_mlu),
                out=self.convert_to_channel_last(out_mlu),
            )
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_select_exception(self):
        x = torch.rand((1, 6), device="mlu").expand((2, 6))
        y = torch.rand((3, 6), device="mlu")
        ind = torch.tensor([0, 1], dtype=torch.int64, device="mlu")
        ref_msg = r"unsupported operation: more than one element of the written-to tensor refers "
        ref_msg += r"to a single memory location\. Please clone\(\) the tensor before performing "
        ref_msg += r"the operation."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.index_select(y, 1, ind, out=x)

        shape = (
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            6,
        )
        x = torch.rand(shape, device="mlu")
        ref_msg = r"Tensor too large or too many \(\> 8\) dimensions"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.index_select(x, 1, ind)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_index_select_bfloat16(self):
        shape_list = [
            (2, 4, 5),
        ]
        c_lists = [
            10,
        ]
        index_shape_list = [
            (3,),
        ]
        type_list = [
            torch.bfloat16,
        ]
        for shape, t, index_shape in product(shape_list, type_list, index_shape_list):
            x = torch.testing.make_tensor(shape, dtype=t, device="cpu")
            index = torch.randint(0, 3, (index_shape)).long()
            if t.is_floating_point:
                x_cpu = torch.nn.Parameter(x)
                x_mlu = torch.nn.Parameter(x.mlu())
            else:
                x_cpu = x
                x_mlu = x.mlu()
            out_cpu = torch.index_select(x_cpu, 1, index)
            out_mlu = torch.index_select(x_mlu, 1, index.mlu())
            self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0.0)
            if t.is_floating_point:
                grad_cpu = torch.randn(out_cpu.size())
                grad_mlu = grad_cpu.to("mlu")
                out_cpu.backward(grad_cpu)
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_cpu.grad.float(), x_mlu.grad.cpu().float(), 0.03
                )


if __name__ == "__main__":
    run_tests()
