from __future__ import print_function
import logging
import sys
import os

import unittest
from itertools import product

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    read_card_info,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()


class TestCatOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_cat_type(self):
        in_shape1 = (1, 2, 3)
        in_shape2 = (1, 77, 3)
        dtypes = [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int,
            torch.long,
            torch.bool,
            torch.float,
            torch.double,
            torch.half,
            torch.complex32,
            torch.complex64,
            torch.complex128,
        ]
        for dtype in dtypes:
            input1 = torch.ones(in_shape1, dtype=dtype)
            input2 = torch.ones(in_shape2, dtype=dtype)
            inputs_cpu = [input1, input2]
            inputs_mlu = [
                self.to_mlu_dtype(input1, dtype),
                self.to_mlu_dtype(input2, dtype),
            ]

            output_cpu = torch.cat(inputs_cpu, dim=1)
            output_mlu = torch.cat(inputs_mlu, dim=1)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_empty(self):
        in_shapes1 = [(0, 3, 32, 32), (4, 3, 32, 32)]
        in_shapes2 = [(0), (4, 32, 32)]
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for [in_shape1, in_shape2] in [in_shapes1, in_shapes2]:
            for dtype, err in dtypes:
                input1 = torch.randn(in_shape1, dtype=torch.float)
                input2 = torch.randn(in_shape2, dtype=torch.float)

                inputs_cpu = [input1, input2]
                inputs_mlu = [
                    self.to_mlu_dtype(input1, dtype),
                    self.to_mlu_dtype(input2, dtype),
                ]

                output_cpu = torch.cat(inputs_cpu, dim=0)
                output_mlu = torch.cat(inputs_mlu, dim=0)
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_channel_last(self):
        in_shapes1 = [(4, 5, 32, 32), (4, 3, 32, 32)]
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for [in_shape1, in_shape2] in [in_shapes1]:
            for dtype, err in dtypes:
                input1 = torch.randn(in_shape1, dtype=torch.float)
                input2 = torch.randn(in_shape2, dtype=torch.float)
                input_channels_last = self.convert_to_channel_last(input1)
                inputs_cpu = [input1, input2]
                inputs_mlu = [
                    self.to_mlu_dtype(input_channels_last, dtype),
                    self.to_mlu_dtype(input2, dtype),
                ]

                output_cpu = torch.cat(inputs_cpu, dim=1)
                output_mlu = torch.cat(inputs_mlu, dim=1)
                output_mlu_channels_first = output_mlu.cpu().float().contiguous()
                self.assertTensorsEqual(output_cpu, output_mlu_channels_first, err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_not_dense(self):
        in_shapes1 = [(4, 5, 32, 32), (4, 3, 32, 32)]
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for [in_shape1, in_shape2] in [in_shapes1]:
            for dtype, err in dtypes:
                input1 = torch.randn(in_shape1, dtype=torch.float)
                input2 = torch.randn(in_shape2, dtype=torch.float)
                inputs_cpu = [input1[:, :15], input2[:, :15]]
                inputs_mlu = [
                    self.to_mlu_dtype(input1, dtype)[:, :15],
                    self.to_mlu_dtype(input2, dtype)[:, :15],
                ]

                output_cpu = torch.cat(inputs_cpu, dim=1)
                output_mlu = torch.cat(inputs_mlu, dim=1)
                output_mlu_channels_first = output_mlu.cpu().float().contiguous()
                self.assertTensorsEqual(output_cpu, output_mlu_channels_first, err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_out(self):
        dtypes = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in dtypes:
            x = torch.randn((24,), dtype=torch.float)
            x_mlu = self.to_mlu_dtype(x.clone(), data_type)

            out_cpu = torch.randn((4,), dtype=torch.float)
            out_mlu = self.to_mlu_dtype(torch.randn((4,), dtype=torch.float), data_type)

            torch.cat([x[:2], x[4:6]], out=out_cpu)
            torch.cat([x_mlu[:2], x_mlu[4:6]], out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_promote(self):
        in_shapes1 = [(0, 3, 32, 32), (4, 3, 32, 32), (6, 3, 32, 32)]
        in_shapes2 = [(0), (1, 3, 32, 32), (1, 3, 32, 32)]
        in_shapes3 = [(4, 3, 32, 32), (1, 3, 32, 32), (2, 3, 32, 32)]
        in_shapes4 = [(0), (0), (0)]
        dtypes1 = [
            torch.double,
            torch.int8,
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.uint8,
        ]
        list_list = [dtypes1, dtypes1, dtypes1]
        for [in_shape1, in_shape2, in_shape3] in [
            in_shapes1,
            in_shapes2,
            in_shapes3,
            in_shapes4,
        ]:
            for dtype1, dtype2, dtype3 in product(*list_list):
                input1 = torch.randn(in_shape1, dtype=torch.float).to(dtype1)
                input2 = torch.randn(in_shape2, dtype=torch.float).to(dtype2)
                input3 = torch.randn(in_shape3, dtype=torch.float).to(dtype3)

                inputs_cpu = [input1, input2, input3]
                inputs_mlu = [input1.mlu(), input2.mlu(), input3.mlu()]

                output_cpu = torch.cat(inputs_cpu, dim=0)
                output_mlu = torch.cat(inputs_mlu, dim=0)
                self.assertTrue(output_cpu.dtype == output_mlu.dtype)
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-3
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_out_promote(self):
        in_shapes1 = [(0, 3, 32, 32), (4, 3, 32, 32), (6, 3, 32, 32)]
        in_shapes2 = [(0), (1, 3, 32, 32), (1, 3, 32, 32)]
        in_shapes3 = [(4, 3, 32, 32), (1, 3, 32, 32), (2, 3, 32, 32)]
        in_shapes4 = [(0), (0), (0)]
        dtypes1 = [
            torch.double,
            torch.int8,
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.uint8,
        ]
        list_list = [dtypes1, dtypes1, dtypes1]
        for [in_shape1, in_shape2, in_shape3] in [
            in_shapes1,
            in_shapes2,
            in_shapes3,
            in_shapes4,
        ]:
            for dtype1, dtype2, dtype3 in product(*list_list):
                input1 = torch.randn(in_shape1, dtype=torch.float).to(dtype1)
                input2 = torch.randn(in_shape2, dtype=torch.float).to(dtype2)
                input3 = torch.randn(in_shape3, dtype=torch.float).to(dtype3)
                inputs_cpu = [input1, input2, input3]
                inputs_mlu = [input1.mlu(), input2.mlu(), input3.mlu()]

                output_cpu = torch.randn((4, 4, 4), dtype=torch.float)
                output_mlu = output_cpu.mlu()
                torch.cat(inputs_cpu, dim=0, out=output_cpu)
                torch.cat(inputs_mlu, dim=0, out=output_mlu)
                self.assertTrue(output_cpu.dtype == output_mlu.dtype)
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-3
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("58GB")
    def test_cat_large(self):
        in_shape1 = (5, 1024, 1024, 1024)
        in_shape2 = (5, 1024, 1024, 1024)
        dtypes = [torch.half, torch.int8]
        for dtype in dtypes:
            input1 = torch.ones(in_shape1, dtype=torch.float).to(dtype)
            input2 = torch.ones(in_shape2, dtype=torch.float).to(dtype)

            inputs_cpu = [input1, input2]
            inputs_mlu = [input1.mlu(), input2.mlu()]

            output_cpu = torch.cat(inputs_cpu, dim=1)
            output_mlu = torch.cat(inputs_mlu, dim=1)
            self.assertTrue(output_cpu.dtype == output_mlu.dtype)
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 3e-3)

    # @unittest.skip("not test")
    @testinfo()
    def test_cat_zero_dim(self):
        input1 = torch.randn(0, dtype=torch.float32, requires_grad=True)
        input2 = torch.randn(0, dtype=torch.float32, requires_grad=True)
        input1_mlu = torch.randn(0, dtype=torch.float32).mlu().requires_grad_(True)
        input2_mlu = torch.randn(0, dtype=torch.float32).mlu().requires_grad_(True)
        inputs_mlu = [input1_mlu, input2_mlu]
        inputs_cpu = [input1, input2]
        output_mlu = torch.cat(inputs_mlu, dim=0)
        output_cpu = torch.cat(inputs_cpu, dim=0)
        output_cpu.backward(torch.randn_like(output_cpu.data))
        output_mlu.backward(torch.randn_like(output_mlu.data))
        self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0.0)
        self.assertTensorsEqual(input1.grad, input1_mlu.grad, 0.0)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_cat_bfloat16(self):
        in_shape1 = (1, 2, 3)
        in_shape2 = (1, 77, 3)
        dtype_list = [
            (torch.float, torch.bfloat16, 3e-3),
            (torch.half, torch.bfloat16, 3e-3),
            (torch.int, torch.bfloat16, 3e-3),
            (torch.short, torch.bfloat16, 3e-3),
            (torch.int8, torch.bfloat16, 3e-3),
            (torch.uint8, torch.bfloat16, 3e-3),
            (torch.double, torch.bfloat16, 3e-3),
            (torch.long, torch.bfloat16, 3e-3),
            (torch.bfloat16, torch.bfloat16, 0),
        ]
        for dtype in dtype_list:
            input1 = torch.ones(in_shape1, dtype=dtype[0])
            input2 = torch.ones(in_shape2, dtype=dtype[1])
            input2_cpu = torch.nn.Parameter(input2)
            input2_mlu = torch.nn.Parameter(input2.mlu())
            inputs_cpu = [input1, input2_cpu]
            inputs_mlu = [input1.mlu(), input2_mlu]

            output_cpu = torch.cat(inputs_cpu, dim=1)
            output_mlu = torch.cat(inputs_mlu, dim=1)
            grad = torch.randn_like(output_cpu)
            output_cpu.backward(grad)
            output_mlu.backward(grad.mlu())
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
            self.assertTensorsEqual(input2_cpu.grad, input2_mlu.grad.cpu(), 0)


if __name__ == "__main__":
    run_tests()
