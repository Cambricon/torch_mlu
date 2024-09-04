from __future__ import print_function

import sys
import os
import unittest
import logging
import math
from itertools import product
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

trigon_list = [
    [torch.sin, -2 * math.pi, 2 * math.pi, "sin"],
    [torch.asin, -0.9, 0.9, "asin"],
    [torch.sinh, -4.1815, 4.1815, "sinh"],
    # [torch.asinh, -100.0, 100.0, "asinh"],
    [torch.cos, -2 * math.pi, 2 * math.pi, "cos"],
    [torch.acos, -0.9, 0.9, "acos"],
    [torch.cosh, -4.8749, 4.8749, "cosh"],
    [torch.acosh, 1.001, 100.0, "acosh"],
    [torch.atan, -1, -1, "atan"],
    [torch.atanh, -0.8, 0.8, "atanh"],
    [torch.tan, -1, 1, "tan"],
]

trigon_list_ = [
    ["sin", -2 * math.pi, 2 * math.pi, "sin"],
    ["asin", -0.9, 0.9, "asin"],
    ["sinh", -4.1815, 4.1815, "sinh"],
    # ["asinh", -100.0, 100.0, "asinh"],
    ["cos", -2 * math.pi, 2 * math.pi, "cos"],
    ["acos", -0.9, 0.9, "acos"],
    ["cosh", -4.8749, 4.8749, "cosh"],
    ["acosh", 1.001, 100.0, "acosh"],
    ["atan", -1, -1, "atan"],
    ["atanh", -0.8, 0.8, "atanh"],
    ["tan", -1, 1, "tan"],
]


def trigon_inplace(tensor, trigon):
    if trigon == "sin":
        tensor.sin_()
    elif trigon == "asin":
        tensor.asin_()
    elif trigon == "sinh":
        tensor.sinh_()
    elif trigon == "asinh":
        tensor.asinh_()
    elif trigon == "cos":
        tensor.cos_()
    elif trigon == "acos":
        tensor.acos_()
    elif trigon == "cosh":
        tensor.cosh_()
    elif trigon == "acosh":
        tensor.acosh_()
    elif trigon == "tan":
        tensor.tan_()
    elif trigon == "atan":
        tensor.atan_()
    elif trigon == "tanh":
        tensor.tanh_()
    elif trigon == "atanh":
        tensor.atanh_()
    return tensor


class TestTrigonOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_trigon(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 30, 40)]
        type_list = [[torch.float, 3e-3], [torch.half, 1e-2], [torch.double, 3e-3]]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type_err, func, trigon in product(
            shape_list, type_list, func_list, trigon_list
        ):
            if trigon[1] == trigon[2] == -1:
                x = torch.randn(shape, dtype=type_err[0])
            else:
                x = torch.empty(shape, dtype=type_err[0]).uniform_(trigon[1], trigon[2])
            input_mlu = self.to_mlu(x)
            input_cpu = func(x)
            input_mlu = func(input_mlu)
            x_cpu = trigon[0](input_cpu.float())
            x_mlu = trigon[0](input_mlu)
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset(), trigon[3])
            self.assertTensorsEqual(
                x_cpu, x_mlu.cpu().float(), type_err[1], use_MSE=True, message=trigon[3]
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_trigon_inplace(self):
        shape_list = [(12, 8, 6, 1), (254, 254, 32, 1, 1, 3)]
        type_list = [[torch.float, 3e-3], [torch.half, 1e-2], [torch.double, 3e-3]]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type_err, func, trigon_ in product(
            shape_list, type_list, func_list, trigon_list_
        ):
            if trigon_[1] == trigon_[2] == -1:
                x_cpu = torch.randn(shape, dtype=type_err[0])
            else:
                x_cpu = torch.empty(shape, dtype=type_err[0]).uniform_(
                    trigon_[1], trigon_[2]
                )
            x_mlu = self.to_mlu(x_cpu)
            x_cpu = func(x_cpu.float())
            x_mlu = func(x_mlu)
            x_ptr = x_mlu.data_ptr()
            x_cpu = trigon_inplace(x_cpu, trigon_[0])
            x_mlu = trigon_inplace(x_mlu, trigon_[0])
            self.assertEqual(x_ptr, x_mlu.data_ptr(), trigon_[3])
            self.assertTrue(x_cpu.stride() == x_mlu.stride(), trigon_[3])
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(
                x_cpu.float(),
                x_mlu.cpu().float(),
                type_err[1],
                use_MSE=True,
                message=trigon_[3],
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.preserve_format),
                x_cpu.is_contiguous(memory_format=torch.preserve_format),
                msg=trigon_[3],
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.channels_last),
                x_cpu.is_contiguous(memory_format=torch.channels_last),
                msg=trigon_[3],
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.contiguous_format),
                x_cpu.is_contiguous(memory_format=torch.contiguous_format),
                msg=trigon_[3],
            )

    # TODO(xuijian1): Need fix pipeline random failure
    @unittest.skip("not test")
    @testinfo()
    def test_trigon_out(self):
        shape_list = [(128, 20, 32, 4, 2), (2, 3, 4, 5), (1, 2, 3, 4)]
        out_shape_list = [(), (1), (1, 2, 3, 4)]
        type_list = [[torch.float, 3e-3], [torch.half, 1e-2], [torch.double, 3e-3]]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, out_shape, type_err, func, func_out, trigon in product(
            shape_list, out_shape_list, type_list, func_list, func_list, trigon_list
        ):
            if trigon[1] == trigon[2] == -1:
                x = torch.randn(shape, dtype=type_err[0])
            else:
                x = torch.empty(shape, dtype=type_err[0]).uniform_(trigon[1], trigon[2])
            out_cpu = torch.randn(out_shape)
            x_mlu = self.to_mlu(x)
            x = func(x)
            x_mlu = func(x_mlu)
            out_mlu = torch.randn(out_shape, dtype=type_err[0]).mlu()
            out_cpu = func_out(out_cpu)
            out_mlu = func_out(out_mlu)
            trigon[0](x.float(), out=out_cpu)
            trigon[0](x_mlu, out=out_mlu)
            self.assertTensorsEqual(
                out_cpu.float(),
                out_mlu.cpu().float(),
                type_err[1],
                use_MSE=True,
                message=trigon[3],
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_trigon_special_case(self):
        shape_list = [(0, 6), ()]
        type_list = [[torch.float, 3e-3], [torch.half, 1e-2], [torch.double, 3e-3]]
        for shape, type_err, trigon in product(shape_list, type_list, trigon_list):
            if trigon[1] == trigon[2] == -1:
                x = torch.randn(shape, dtype=type_err[0])
            else:
                x = torch.empty(shape, dtype=type_err[0]).uniform_(trigon[1], trigon[2])
            x_cpu = trigon[0](x.float())
            x_mlu = trigon[0](self.to_mlu(x))
            self.assertTrue(x_cpu.stride() == x_mlu.stride(), trigon[3])
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset(), trigon[3])
            self.assertTensorsEqual(
                x_cpu, x_mlu.cpu().float(), type_err[1], use_MSE=True, message=trigon[3]
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.preserve_format),
                x_cpu.is_contiguous(memory_format=torch.preserve_format),
                msg=trigon[3],
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.channels_last),
                x_cpu.is_contiguous(memory_format=torch.channels_last),
                msg=trigon[3],
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.contiguous_format),
                x_cpu.is_contiguous(memory_format=torch.contiguous_format),
                msg=trigon[3],
            )

        shape_out_list = [(), (5, 0)]
        shape_list = [(2), (0, 6), ()]
        for shape, shape_out, type_err, trigon in product(
            shape_list, shape_out_list, type_list, trigon_list
        ):
            if trigon[1] == trigon[2] == -1:
                x = torch.randn(shape, dtype=type_err[0])
            else:
                x = torch.empty(shape, dtype=type_err[0]).uniform_(trigon[1], trigon[2])
            out_cpu = torch.randn(shape_out)
            out_mlu = torch.randn(shape_out, dtype=type_err[0]).mlu()
            trigon[0](x.float(), out=out_cpu)
            trigon[0](self.to_mlu(x), out=out_mlu)
            self.assertTrue(out_cpu.stride() == out_mlu.stride(), trigon[3])
            self.assertTrue(
                out_mlu.storage_offset() == out_cpu.storage_offset(), trigon[3]
            )
            self.assertTensorsEqual(
                out_cpu,
                out_mlu.cpu().float(),
                type_err[1],
                use_MSE=True,
                message=trigon[3],
            )
            self.assertEqual(
                out_mlu.is_contiguous(memory_format=torch.preserve_format),
                out_cpu.is_contiguous(memory_format=torch.preserve_format),
                msg=trigon[3],
            )
            self.assertEqual(
                out_mlu.is_contiguous(memory_format=torch.channels_last),
                out_cpu.is_contiguous(memory_format=torch.channels_last),
                msg=trigon[3],
            )
            self.assertEqual(
                out_mlu.is_contiguous(memory_format=torch.contiguous_format),
                out_cpu.is_contiguous(memory_format=torch.contiguous_format),
                msg=trigon[3],
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_trigon_permute(self):
        type_list = [[torch.float, 3e-3], [torch.half, 1e-2], [torch.double, 3e-3]]
        shape_list = [
            [(512, 1024, 2, 2, 4), (2, 4, 3, 1, 0)],
            [(2, 3, 4), (2, 1, 0)],
            [(254, 254, 112, 1, 1, 3), (4, 2, 5, 3, 1, 0)],
            [(0, 6), (1, 0)],
        ]
        for shape, type_err, trigon in product(shape_list, type_list, trigon_list):
            if trigon[1] == trigon[2] == -1:
                x = torch.randn(shape[0], dtype=type_err[0]).permute(shape[1])
            else:
                x = (
                    torch.empty(shape[0], dtype=type_err[0])
                    .uniform_(trigon[1], trigon[2])
                    .permute(shape[1])
                )
            out_cpu = trigon[0](x.float())
            out_mlu = trigon[0](self.to_mlu(x))
            self.assertTensorsEqual(
                out_cpu,
                out_mlu.cpu().float(),
                type_err[1],
                use_MSE=True,
                message=trigon[3],
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("42GB")
    def test_trigon_atan2_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        type_list = [[torch.half, 1e-2]]
        for shape, type_err in product(shape_list, type_list):
            x = torch.randn(shape, dtype=type_err[0])
            y = torch.randn(1024, dtype=type_err[0])
            x_mlu, y_mlu = self.to_mlu(x), self.to_mlu(y)
            x_cpu = torch.atan2(x.float(), y.float())
            x_mlu = torch.atan2(x_mlu, y_mlu)
            self.assertTensorsEqual(
                x_cpu, x_mlu.cpu().float(), type_err[1], use_MSE=True, message="atan2"
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_trigon_large(self):
        trigon_large_list = [
            [torch.asin, -0.9, 0.9, "asin"],
            [torch.acos, -0.9, 0.9, "acos"],
            [torch.acosh, 1.001, 100.0, "acosh"],
            [torch.atan, -1, -1, "atan"],
            [torch.atanh, -0.8, 0.8, "atanh"],
            [torch.tan, -1, 1, "tan"],
            [torch.sin, -2 * math.pi, 2 * math.pi, "sin"],
            [torch.cos, -2 * math.pi, 2 * math.pi, "cos"],
        ]
        shape_list = [(5, 1024, 1024, 1024)]
        type_list = [[torch.half, 1e-2]]
        for shape, type_err, trigon in product(shape_list, type_list, trigon_list):
            if trigon[1] == trigon[2] == -1:
                x = torch.randn(shape, dtype=type_err[0])
            else:
                x = torch.empty(shape, dtype=type_err[0]).uniform_(trigon[1], trigon[2])
            x_cpu = trigon[0](x.float())
            x_mlu = trigon[0](self.to_mlu(x))
            self.assertTensorsEqual(
                x_cpu, x_mlu.cpu().float(), type_err[1], use_MSE=True, message=trigon[3]
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_trigon_bfloat16(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 30, 40)]
        type_list = [
            [torch.bfloat16, 3e-3],
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for shape, type_err, func, trigon in product(
            shape_list, type_list, func_list, trigon_list
        ):
            if trigon[1] == trigon[2] == -1:
                x = torch.randn(shape, dtype=type_err[0])
            else:
                x = torch.empty(shape, dtype=type_err[0]).uniform_(trigon[1], trigon[2])
            input_mlu = self.to_mlu(x)
            input_cpu = func(x)
            input_mlu = func(input_mlu)
            x_cpu = trigon[0](input_cpu.float())
            x_mlu = trigon[0](input_mlu)
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset(), trigon[3])
            self.assertTensorsEqual(
                x_cpu, x_mlu.cpu().float(), type_err[1], use_MSE=True, message=trigon[3]
            )


if __name__ == "__main__":
    run_tests()
