from __future__ import print_function
import os

import sys
import logging
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
    TEST_BFLOAT16,
    largeTensorTest,
)  # pylint: disable=C0413,C0411
from torch.testing import make_tensor

logging.basicConfig(level=logging.DEBUG)


class TestExpandOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_expand(self):
        shape_list = [
            ((), (5, 8)),
            ((), (2, 3, 4, 5)),
            ((5), (256, 5)),
            ((4, 5), (1, 3, 4, 5)),
            ((4, 5), (0, 4, 5)),
            ((2, 3, 4), (3, -1, 3, 4)),
            ((128, 1, 1024), (-1, 379, -1)),
            ((2048, 5), (1, 3, 2048, 5)),
            ((24, 1, 1, 1), (24, 51, 51, 1)),
            ((2, 6, 1, 1), (24, 51, 2, 6, 1, 1)),
            ((7, 56), (256, 0, 7, 56)),
            ((2048, 5), (0, 3, 2048, 5)),
            ((8, 1, 64, 64), (8, 512, 64, 64)),
            ((8, 2, 64, 64), (8, 2, 64, 64)),
            ((8, 2, 64, 64), (1, 8, 2, 64, 64)),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, func in product(shape_list, func_list):
            shape1, shape2 = shape
            x = torch.randn(shape1, dtype=torch.float)
            x_mlu = x.to("mlu")
            out_cpu = func(x).expand(shape2) * 3.14
            out_mlu = func(x_mlu).expand(shape2) * 3.14
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_expand_with_channels_last(self):
        for shape1, shape2 in [
            ((), (5, 8)),
            ((), (2, 3, 4, 5)),
            ((5), (256, 5)),
            ((4, 5), (1, 3, 4, 5)),
            ((4, 5), (0, 4, 5)),
            ((2, 3, 4), (3, -1, 3, 4)),
            ((128, 1, 1024), (-1, 379, -1)),
            ((2048, 5), (1, 3, 2048, 5)),
            ((24, 1, 1, 1), (24, 51, 51, 1)),
            ((2, 6, 1, 1), (24, 51, 2, 6, 1, 1)),
            ((7, 56), (256, 0, 7, 56)),
            ((2048, 5), (0, 3, 2048, 5)),
            ((8, 1, 64, 64), (8, 512, 64, 64)),
            ((8, 2, 64, 64), (8, 2, 64, 64)),
            ((8, 2, 64, 64), (1, 8, 2, 64, 64)),
        ]:
            x = torch.randn(shape1, dtype=torch.float)
            out_cpu = x.expand(shape2) * 3.14
            out_mlu = self.to_mlu(x).expand(shape2) * 3.14
            if out_mlu.dim() == 4:
                out_cpu = out_cpu.contiguous(memory_format=torch.channels_last)
                out_mlu = out_mlu.contiguous(memory_format=torch.channels_last)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_expand_exception(self):
        a = torch.randn((2, 2, 3, 3), dtype=torch.float).to("mlu")
        ref_msg = (
            r"^expand\(mluFloatType\{\[2, 2, 3, 3\]\}, size=\[2, 3, 3\]\): the number"
        )
        ref_msg = (
            ref_msg + r" of sizes provided \(3\) must be greater or equal to the number"
        )
        ref_msg = ref_msg + r" of dimensions in the tensor \(4\)$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.expand((2, 3, 3))

    @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "only run case of large tensor by `--large`")
    @largeTensorTest("46GB")
    def test_expand_large(self):
        for shape1, shape2 in [((5), (1024, 1024, 1024, 5))]:
            x = torch.randn(shape1, dtype=torch.float)
            out_cpu = x.expand(shape2)
            out_mlu = self.to_mlu(x).expand(shape2)
            if out_mlu.dim() == 4:
                out_cpu = out_cpu.contiguous(memory_format=torch.channels_last)
                out_mlu = out_mlu.contiguous(memory_format=torch.channels_last)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_expand_bfloat16(self):
        input_t = torch.rand((2, 1, 24, 1), dtype=torch.bfloat16)
        input_cpu = torch.nn.Parameter(input_t)
        input_mlu = torch.nn.Parameter(input_t.mlu())
        out_cpu = input_cpu.expand((2, 4, 24, 5))
        out_mlu = input_mlu.expand((2, 4, 24, 5))
        grad = make_tensor(
            out_cpu.size(), dtype=torch.bfloat16, device="cpu", low=-1, high=1
        )
        out_cpu.backward(grad)
        out_mlu.backward(grad.mlu())
        self.assertEqual(out_cpu, out_mlu.cpu())
        self.assertEqual(input_cpu.grad, input_mlu.grad.cpu(), atol=0.01, rtol=0.016)


if __name__ == "__main__":
    run_tests()
