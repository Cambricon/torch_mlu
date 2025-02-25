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
    TEST_BFLOAT16,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0411, C0413

logging.basicConfig(level=logging.DEBUG)


class TestLeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_le(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
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
                out_cpu = torch.le(x, y)
                out_mlu = torch.le(self.to_mlu(x), self.to_mlu(y))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

                out_cpu = x <= y
                out_mlu = self.to_mlu(x) <= self.to_mlu(y)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_not_dense(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
        ]
        for t in type_list:
            for shape1, shape2 in [
                ((12, 15, 18, 26), (12, 15, 18, 50)),
                ((1), (20, 144, 8, 30)),
            ]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                out_cpu = torch.le(x, y[:, :, :, 10:36])
                y_mlu = self.to_mlu(y)
                out_mlu = torch.le(self.to_mlu(x), y_mlu[:, :, :, 10:36])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

                out_cpu = x <= y[:, :, :, 10:36]
                out_mlu = self.to_mlu(x) <= y_mlu[:, :, :, 10:36]
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_channel_last(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
        ]
        for t in type_list:
            for shape1 in [(12, 15, 18, 26), (20, 144, 8, 30)]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape1).to(t)
                x_cl = self.convert_to_channel_last(x)
                y_cl = self.convert_to_channel_last(y)
                out_cpu = torch.le(x_cl, y_cl)
                x_mlu_cl = self.convert_to_channel_last(self.to_mlu(x))
                y_mlu_cl = self.convert_to_channel_last(self.to_mlu(y))
                out_mlu = torch.le(x_mlu_cl, y_mlu_cl)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

                out_cpu = x_cl <= y_cl
                out_mlu = x_mlu_cl <= y_mlu_cl
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

                # mixed memory format
                z = torch.randn(shape1).to(t)
                out_cpu = torch.le(x, z)
                out_mlu = torch.le(self.to_mlu(x), self.to_mlu(z))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                out_cpu = x <= z
                out_mlu = self.to_mlu(x) <= self.to_mlu(z)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_inplace(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
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
                y.le_(x)
                y_mlu.le_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(
                    y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_inplace_channel_last(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.long,
            torch.half,
            torch.uint8,
        ]
        for t in type_list:
            for shape1, shape2 in [((5, 3, 4, 1), (1, 3, 4, 1))]:
                # both channel last
                x = torch.randn(shape2).to(t).to(memory_format=torch.channels_last)
                y = torch.randn(shape1).to(t).to(memory_format=torch.channels_last)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y.le_(x)
                y_mlu_data = y_mlu.data_ptr()
                y_mlu.le_(x_mlu)
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(
                    y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True
                )
                # mixed memory format
                z = torch.randn(shape1).to(t)
                z_mlu = z.to("mlu")
                z.le_(x)
                z_mlu_data = z_mlu.data_ptr()
                z_mlu.le_(x_mlu)
                self.assertEqual(z_mlu_data, z_mlu.data_ptr())
                self.assertTensorsEqual(
                    z.float(), z_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_inplace_not_dense(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.long,
            torch.half,
            torch.uint8,
        ]
        for t in type_list:
            for shape1, shape2 in [((3, 4), (2, 3, 4))]:
                x = torch.randn(shape1).to(t)
                y = torch.randn(shape2).to(t)
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                y[:, :, :2].le_(x[:, :2])
                y_mlu_data = y_mlu.data_ptr()
                y_mlu[:, :, :2].le_(x_mlu[:, :2])
                self.assertEqual(y_mlu_data, y_mlu.data_ptr())
                self.assertTensorsEqual(
                    y.float(), y_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_out(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
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
                out_tmpcpu = torch.zeros(shape2, dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape2, dtype=torch.bool).to("mlu")
                torch.le(x, y, out=out_tmpcpu)
                torch.le(self.to_mlu(x), self.to_mlu(y), out=out_tmpmlu)
                self.assertTensorsEqual(
                    out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_scalar(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
        ]
        for t in type_list:
            for shape in [
                (),
                (256, 144, 7, 15, 2, 1),
                (1),
                (256, 7),
                (2, 3, 4),
                (117, 1, 5, 1, 5, 1, 3, 1),
                (256, 144, 7, 15, 2, 1, 1, 1),
            ]:
                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                out_cpu = torch.le(x, y)
                out_mlu = torch.le(self.to_mlu(x), y)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                out_cpu = x <= y
                out_mlu = self.to_mlu(x) <= y
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_inplace_scalar(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
        ]
        for t in type_list:
            for shape in [
                (),
                (256, 144, 7, 15, 2, 1),
                (1,),
                (256, 7),
                (2, 3, 4),
                (117, 1, 5, 1, 5, 1, 3, 1),
                (256, 144, 7, 15, 2, 1, 1, 1),
            ]:
                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                x_mlu = x.to("mlu")
                x_mlu_data = x_mlu.data_ptr()
                x.le_(y)
                x_mlu.le_(y)
                self.assertEqual(x_mlu_data, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_out_scalar(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
        ]
        for t in type_list:
            for shape in [
                (),
                (256, 144, 7, 15, 2, 1),
                (1),
                (256, 7),
                (2, 3, 4),
                (117, 1, 5, 1, 5, 1, 3, 1),
                (256, 144, 7, 15, 2, 1, 1, 1),
            ]:
                x = torch.randn(shape).to(t)
                y = torch.randn(()).to(t).item()
                out_tmpcpu = torch.zeros(shape, dtype=torch.bool)
                out_tmpmlu = torch.zeros(shape, dtype=torch.bool).to("mlu")
                torch.le(x, y, out=out_tmpcpu)
                torch.le(self.to_mlu(x), y, out=out_tmpmlu)
                self.assertTensorsEqual(
                    out_tmpcpu.float(), out_tmpmlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_le_and_le_out_after_permute(self):
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
            ouput = torch.le(a_permute, b_permute)
            a_mlu = torch.permute(a.to("mlu"), tuple(permute_size))
            b_mlu = torch.permute(b.to("mlu"), tuple(permute_size))
            ouput_mlu = torch.le(a_mlu, b_mlu)
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 0, use_MSE=True)

            a_permute.le_(b_permute)
            a_mlu.le_(b_mlu)
            self.assertTensorsEqual(a_permute, a_mlu.cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_le_mixed_type(self):
        input1_types = [torch.int, torch.long]
        input2_types = [torch.double, torch.float]
        shapes = [((), ()), ((), (1)), ((2, 3, 4, 6), (3, 4, 6))]
        product_list = product(input1_types, input2_types, shapes)
        for input1_type, input2_type, shape in product_list:
            shape1, shape2 = shape
            a = torch.randn(shape1, dtype=torch.float).to(input1_type)
            b = torch.randn(shape2, dtype=torch.float).to(input2_type)

            ouput = torch.le(a, b)
            ouput_mlu = torch.le(a.mlu(), b.mlu())
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_le_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 3, 4, 6), dtype=torch.bfloat16, device="cpu"
        )
        right = torch.testing.make_tensor((3, 4, 6), dtype=torch.bfloat16, device="cpu")
        out_cpu = torch.le(left, right)
        out_mlu = torch.le(left.mlu(), right.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_le_large(self):
        left = torch.testing.make_tensor(
            (2, 1024, 1024, 1024), dtype=torch.float, device="cpu"
        )
        right = torch.testing.make_tensor(
            (2, 1024, 1024, 1024), dtype=torch.float, device="cpu"
        )
        out_cpu = torch.le(left, right)
        out_mlu = torch.le(left.mlu(), right.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
