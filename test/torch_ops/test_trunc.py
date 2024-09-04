# pylint: disable=W0223,R0201,C0413,C0411,C0301
from __future__ import print_function
from itertools import product

import sys
import os
import unittest
import logging
import copy
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
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()


class TestTruncOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_trunc(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [(torch.float), (torch.half)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype, shape, func in product(dtype_list, shape_list, func_list):
            input = torch.randn(shape, dtype=dtype)
            input_mlu = input.mlu()
            out_cpu = torch.trunc(func(input.float()))
            out_mlu = torch.trunc(func(input_mlu))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_trunc_inplace(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [(torch.float), (torch.half)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype, shape, func in product(dtype_list, shape_list, func_list):
            input = torch.randn(shape, dtype=dtype)
            input_mlu = func(input.mlu())
            input = func(input.float())
            ptr1 = input_mlu.data_ptr()
            input.trunc()
            input_mlu.trunc()
            ptr2 = input_mlu.data_ptr()
            self.assertEqual(ptr1, ptr2)
            self.assertTensorsEqual(input, input_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_trunc_out(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        dtype_list = [(torch.float), (torch.half)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype, shape, func in product(dtype_list, shape_list, func_list):
            input = torch.randn(shape, dtype=dtype)
            input_mlu = func(input.mlu())
            out_cpu = torch.randn(1, dtype=dtype)
            out_mlu = out_cpu.to("mlu")
            out_cpu = out_cpu.float()
            torch.trunc(func(input.float()), out=out_cpu)
            torch.trunc(input_mlu, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_trunc_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.trunc(x, out=out)
            torch.trunc(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_trunc_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x = x.float()
            x.trunc_()
            x_mlu.trunc_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_trunc_backward(self):
        shape_list = [
            (66),
            (39, 48),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (11, 13, 16, 18, 20, 23),
        ]
        type_list = [torch.float]
        for shape in shape_list:
            for data_type in type_list:
                x_0 = torch.randn(shape, dtype=data_type)
                x_mlu = x_0.to("mlu")
                x_0.requires_grad_(True)
                x_mlu.requires_grad_(True)
                out_cpu = torch.trunc(x_0)
                out_mlu = torch.trunc(x_mlu)
                out_cpu.backward(torch.ones_like(out_cpu))
                out_mlu.backward(torch.ones_like(out_mlu))
                self.assertTensorsEqual(
                    x_0.grad, x_mlu.grad.cpu(), 0.003, allow_inf=True, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_trunc_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        dtype_list = [(torch.half)]
        for dtype, shape in product(dtype_list, shape_list):
            input = torch.randn(shape, dtype=dtype)
            input_mlu = input.mlu()
            out_cpu = torch.trunc(input.float())
            out_mlu = torch.trunc(input_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_trunc_bfloat16(self):
        shape_list = [
            (66),
            (39, 48),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (11, 13, 16, 18, 20, 23),
        ]
        type_list = [
            torch.bfloat16,
        ]
        for shape in shape_list:
            for data_type in type_list:
                x_0 = torch.randn(shape, dtype=data_type)
                x_mlu = x_0.to("mlu")
                x_0.requires_grad_(True)
                x_mlu.requires_grad_(True)
                out_cpu = torch.trunc(x_0)
                out_mlu = torch.trunc(x_mlu)
                out_cpu.backward(torch.ones_like(out_cpu))
                out_mlu.backward(torch.ones_like(out_mlu))
                self.assertTensorsEqual(
                    x_0.grad, x_mlu.grad.cpu(), 0.003, allow_inf=True, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
