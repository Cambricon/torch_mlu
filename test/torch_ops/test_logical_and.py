from __future__ import print_function

import sys
import os
import unittest
import logging
import random
from itertools import product
import numpy
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0411, C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestLogicalAndOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_logical_and(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape1, shape2 in [
                ((), ()),
                ((), (1)),
                ((), (256, 144, 7, 15, 2, 1)),
                ((1), (256, 7)),
                ((5), (5)),
                ((2, 3, 4), (3, 4)),
                ((1, 117, 1, 4, 1, 5, 1, 2), (117, 1, 5, 1, 5, 1, 3, 1)),
                ((1), (256, 144, 7, 15, 2, 1, 1, 1)),
            ]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.logical_and(x, y)
                out_mlu = torch.logical_and(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_and_PYTORCH_11152(self):
        x = torch.randn((1, 4, 1, 64, 64))
        y = torch.randn((1, 4, 1, 64, 64))
        x.as_strided_(x.size(), stride=(4, 1, 4, 256, 4))
        y.as_strided_(y.size(), stride=(16384, 1, 16384, 256, 4))
        out_cpu = torch.logical_and(x, y)
        out_mlu = torch.logical_and(self.to_mlu(x), self.to_mlu(y))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_and_not_dense(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape1, shape2 in [
                ((12, 15, 18, 26), (12, 15, 18, 50)),
                ((1), (20, 144, 8, 30)),
            ]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.logical_and(x, y[:, :, :, 10:36])
                y_mlu = self.to_mlu(y)
                out_mlu = torch.logical_and(self.to_mlu(x), y_mlu[:, :, :, 10:36])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_and_channels_last(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape1 in [(12, 15, 18, 26), (20, 144, 8, 30)]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape1).to(t)
                x_cl = self.convert_to_channel_last(x)
                y_cl = self.convert_to_channel_last(y)
                out_cpu = torch.logical_and(x_cl, y_cl)
                x_mlu_cl = self.convert_to_channel_last(self.to_mlu(x))
                y_mlu_cl = self.convert_to_channel_last(self.to_mlu(y))
                out_mlu = torch.logical_and(x_mlu_cl, y_mlu_cl)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_and_inplace(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape1, shape2 in [
                ((), ()),
                ((), (1)),
                ((), (256, 144, 7, 15, 2, 1)),
                ((1), (256, 7)),
                ((5), (5)),
                ((1, 117, 1, 4, 1, 1, 1, 1), (117, 117, 5, 4, 5, 1, 3, 1)),
                ((1), (256, 144, 7, 15, 2, 1, 1, 1)),
            ]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y_mlu_data = y_mlu.data_ptr()
                y.logical_and_(x)
                y_mlu.logical_and_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(
                    y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_and_out(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape1, shape2 in [
                ((), ()),
                ((), (1)),
                ((), (256, 144, 7, 15, 2, 1)),
                ((1), (256, 7)),
                ((5), (5)),
                ((2, 3, 4), (3, 4)),
                ((1, 117, 1, 4, 1, 1, 1, 1), (117, 117, 5, 4, 5, 1, 3, 1)),
                ((256, 144, 7, 15, 2, 1, 1, 1), (1)),
                ((1), (256, 144, 7, 15, 2, 1, 1, 1)),
            ]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                broadcast_shape = torch._C._infer_size(x.shape, y.shape)
                out_cpu = torch.zeros(broadcast_shape, dtype=torch.bool)
                out_mlu = torch.zeros(broadcast_shape, dtype=torch.bool).to("mlu")
                torch.logical_and(x, y, out=out_cpu)
                torch.logical_and(self.to_mlu(x), self.to_mlu(y), out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_and_and_logical_and_out_after_permute(self):
        for i in range(5):
            dimention_list = []
            for k in range(i + 3):  # pylint: disable=W0612
                dimention_list.append(numpy.random.randint(1, 20))
            shape = tuple(dimention_list)
            permute_size = numpy.arange(len(shape))
            random.shuffle(permute_size)

            a = torch.randn(shape)
            b = torch.randn(shape)
            a_permute = torch.permute(a, tuple(permute_size))
            b_permute = torch.permute(b, tuple(permute_size))
            ouput = torch.logical_and(a_permute, b_permute)
            a_mlu = torch.permute(a.to("mlu"), tuple(permute_size))
            b_mlu = torch.permute(b.to("mlu"), tuple(permute_size))
            ouput_mlu = torch.logical_and(a_mlu, b_mlu)
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 0, use_MSE=True)

            a_permute.logical_and_(b_permute)
            a_mlu.logical_and_(b_mlu)
            self.assertTensorsEqual(a_permute, a_mlu.cpu(), 0, use_MSE=True)

    def test_logical_and_mixed_type(self):
        input1_types = [torch.int, torch.long]
        input2_types = [torch.double, torch.float]
        shapes = [((), ()), ((), (1)), ((2, 3, 4, 6), (3, 4, 6))]
        product_list = product(input1_types, input2_types, shapes)
        for input1_type, input2_type, shape in product_list:
            shape1, shape2 = shape
            a = torch.randn(shape1, dtype=torch.float).to(input1_type)
            b = torch.randn(shape2, dtype=torch.float).to(input2_type)

            ouput = torch.logical_and(a, b)
            ouput_mlu = torch.logical_and(a.mlu(), b.mlu())
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_logical_and_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 3, 4, 6), dtype=torch.bfloat16, device="cpu"
        )
        right = torch.testing.make_tensor((3, 4, 6), dtype=torch.bfloat16, device="cpu")
        out_cpu = torch.logical_and(left, right)
        out_mlu = torch.logical_and(left.mlu(), right.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("48GB")
    def test_logical_and_large(self):
        left = torch.testing.make_tensor(
            (2, 1024, 1024, 1024), dtype=torch.float, device="cpu"
        )
        right = torch.testing.make_tensor(
            (2, 1024, 1024, 1024), dtype=torch.float, device="cpu"
        )
        out_cpu = torch.logical_and(left, right)
        out_mlu = torch.logical_and(left.mlu(), right.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
