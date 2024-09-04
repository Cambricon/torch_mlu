from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import random
import numpy
import torch
from torch import nn
from itertools import product

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    read_card_info,
    largeTensorTest,
)

TEST_BFLOAT16 = read_card_info()


class TestSubModel(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(TestSubModel, self).__init__()

    def forward(self, x, y):  # pylint: disable=R0201
        z = x - y
        return z


class TestSubScaleModel(nn.Module):  # pylint: disable=W0223
    def __init__(self, scale, alpha=1.0):
        super(TestSubScaleModel, self).__init__()
        self.scale = scale
        self.alpha = alpha

    def forward(self, x):
        y = x.sub(self.scale, self.alpha)
        return y


class TestSubOp(TestCase):  # pylint: disable=R0904
    # @unittest.skip("not test")
    @testinfo()
    def test_sub(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
        ]
        shape_list = [
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 30, 80), (2, 30, 80)),
            ((3, 20), (3, 20)),
            ((10), (10)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((1, 3, 224, 224), (1, 1, 1, 1)),
            ((1, 3, 224, 224), (1)),
            ((1, 3, 224), (1, 3, 1)),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func in product(shape_list, dtype_list, func_list):
            x_left = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_right = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu = x_left.mlu()
            x_right_mlu = x_right.mlu()
            out_cpu = torch.sub(func(x_left), func(x_right))
            out_mlu1 = torch.sub(func(x_left_mlu), func(x_right_mlu))
            out_mlu2 = torch.subtract(func(x_left_mlu), func(x_right_mlu))
            self.assertEqual(out_cpu.dtype, out_mlu1.dtype)
            self.assertEqual(out_cpu.dtype, out_mlu2.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu1.cpu().float(), dtype_err[1], use_MSE=True
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu2.cpu().float(), dtype_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsub(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
        ]
        shape_list = [
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 30, 80), (2, 30, 80)),
            ((3, 20), (3, 20)),
            ((10), (10)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((1, 3, 224, 224), (1, 1, 1, 1)),
            ((1, 3, 224, 224), (1)),
            ((1, 3, 224), (1, 3, 1)),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func in product(shape_list, dtype_list, func_list):
            x_left = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_right = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu = x_left.mlu()
            x_right_mlu = x_right.mlu()
            out_cpu = torch.rsub(func(x_left), func(x_right))
            out_mlu = torch.rsub(func(x_left_mlu), func(x_right_mlu))
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), dtype_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
        ]
        shape_list = [
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 30, 80), (2, 30, 80)),
            ((3, 20), (3, 20)),
            ((10), (10)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((1, 3, 224, 224), (1, 1, 1, 1)),
            ((1, 3, 224, 224), (1)),
            ((1, 3, 224), (1, 3, 1)),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func in product(shape_list, dtype_list, func_list):
            x_left = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_right = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu1 = func(x_left.mlu())
            x_left_mlu1_dptr = x_left_mlu1.data_ptr()
            x_left_mlu2 = func(x_left.mlu())
            x_left_mlu2_dptr = x_left_mlu2.data_ptr()
            x_right_mlu = x_right.mlu()
            x_left_cpu = func(x_left)
            x_left_cpu.sub_(func(x_right))
            x_left_mlu1.sub_(func(x_right_mlu))
            x_left_mlu2.subtract_(func(x_right_mlu))
            self.assertEqual(x_left.dtype, x_left_mlu1.dtype)
            self.assertEqual(x_left.dtype, x_left_mlu2.dtype)
            self.assertEqual(x_left_mlu1_dptr, x_left_mlu1.data_ptr())
            self.assertEqual(x_left_mlu2_dptr, x_left_mlu2.data_ptr())
            self.assertTensorsEqual(
                x_left_cpu.float(),
                x_left_mlu1.cpu().float(),
                dtype_err[1],
                use_MSE=True,
            )
            self.assertTensorsEqual(
                x_left_cpu.float(),
                x_left_mlu2.cpu().float(),
                dtype_err[1],
                use_MSE=True,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_out(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
        ]
        shape_list = [
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 30, 80), (2, 30, 80)),
            ((3, 20), (3, 20)),
            ((10), (10)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((1, 3, 224, 224), (1, 1, 1, 1)),
            ((1, 3, 224, 224), (1)),
            ((1, 3, 224), (1, 3, 1)),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func in product(shape_list, dtype_list, func_list):
            x_left = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_right = torch.testing.make_tensor(
                shape[0], dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu1 = func(x_left.mlu())
            x_left_mlu2 = func(x_left.mlu())
            x_right_mlu = x_right.mlu()
            x_left_cpu = func(x_left)
            # resize output
            out_cpu = torch.empty((0,), dtype=dtype_err[0])
            out_mlu1 = out_cpu.mlu()
            out_mlu2 = out_cpu.mlu()
            torch.sub(x_left_cpu, func(x_right), out=out_cpu)
            torch.sub(x_left_mlu1, func(x_right_mlu), out=out_mlu1)
            torch.subtract(x_left_mlu2, func(x_right_mlu), out=out_mlu2)
            self.assertEqual(out_cpu.dtype, out_mlu1.dtype)
            self.assertEqual(out_cpu.dtype, out_mlu2.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu1.cpu().float(), dtype_err[1], use_MSE=True
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu2.cpu().float(), dtype_err[1], use_MSE=True
            )
            # using left input as output
            torch.sub(x_left_cpu, func(x_right), out=x_left_cpu)
            torch.sub(x_left_mlu1, func(x_right_mlu), out=x_left_mlu1)
            torch.subtract(x_left_mlu2, func(x_right_mlu), out=x_left_mlu2)
            self.assertEqual(x_left_cpu.dtype, x_left_mlu1.dtype)
            self.assertEqual(x_left_cpu.dtype, x_left_mlu2.dtype)
            self.assertTensorsEqual(
                x_left_cpu.float(),
                x_left_mlu1.cpu().float(),
                dtype_err[1],
                use_MSE=True,
            )
            self.assertTensorsEqual(
                x_left_cpu.float(),
                x_left_mlu2.cpu().float(),
                dtype_err[1],
                use_MSE=True,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_scale(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        model = TestSubScaleModel(0.5, 1.5).float()
        for dtype_err, func in product(dtype_list, func_list):
            x_left = torch.testing.make_tensor(
                (1, 3, 224, 224), dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu = func(x_left.mlu())
            out_cpu = model(func(x_left))
            out_mlu = model(x_left_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), dtype_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_tensor(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0.0)]
        for data_type, err in dtype_list:
            # [res] torch.sub([res,] tensor1, tensor2)
            m1 = torch.randn(100, 100).mlu().to(data_type)
            v1 = torch.randn(100).mlu().to(data_type)

            # contiguous
            res1 = torch.sub(m1[4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(1)):
                res2[i] = m1[4, i] - v1[i]
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

            # non-contiguous
            res1 = torch.sub(m1[:, 4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(0)):
                res2[i] = m1[i, 4] - v1[i]
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_value(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # [res] torch.sub([res,] tensor, value)
            m1 = torch.randn(10, 10).mlu().to(data_type)

            # contiguous
            res1 = m1.clone()
            res1[3].sub_(2)
            res2 = m1.clone()
            for i in range(m1.size(1)):
                res2[3, i] = res2[3, i] - 2
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

            # non-contiguous
            m1 = torch.randn(10, 10)
            m2 = m1.mlu().to(data_type)
            res1 = m2.clone()
            res1_data_ptr1 = res1.data_ptr()
            res1[:, 3].sub_(2)
            res1_data_ptr2 = res1.data_ptr()
            res2 = m2.clone()
            res2_data_ptr1 = res2.data_ptr()
            for i in range(m1.size(0)):
                res2[i, 3] = res2[i, 3] - 2
            res2_data_ptr2 = res2.data_ptr()
            cpu_input = m1.clone()
            cpu_input[:, 3].sub_(2)
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )
            self.assertTensorsEqual(cpu_input, res2.cpu().float(), err, use_MSE=True)
            self.assertEqual(res1_data_ptr1, res1_data_ptr2)
            self.assertEqual(res2_data_ptr1, res2_data_ptr2)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inter_type(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            # inter-type
            m1 = torch.randn(10, 10).mlu().to(data_type)
            res1 = m1 - 3
            res2 = m1 - torch.tensor(3).mlu()
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )
            res1 = 3 - m1
            res2 = torch.tensor(3).mlu() - m1
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_empty(self):
        dtype_list = [torch.float, torch.half]
        shape_list = [(), (1)]
        for data_type in dtype_list:
            # (0d or 1d) - empty
            for shape in shape_list:
                m1 = torch.rand(shape, dtype=torch.float).mlu().to(data_type)
                m2 = torch.tensor([], dtype=torch.float).mlu().to(data_type)
                res = m1 - m2
                self.assertEqual(res.cpu().shape, m2.shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_scalar(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape in [(224), (2, 4, 5, 3), (24, 24)]:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 - b.sum()
                out_mlu = 1.2 - self.to_mlu_dtype(b.sum(), data_type)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )
                c = torch.rand(shape, dtype=torch.float)
                out_cpu = c.sum() - 1.2
                out_mlu = self.to_mlu_dtype(c.sum(), data_type) - 1.2
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_batchnorm_sub(self):
        b = torch.tensor(100, dtype=torch.long)
        out_cpu = b - 1
        out_mlu = b.mlu() - 1
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)
        c = torch.tensor(100, dtype=torch.long)
        out_cpu = 1 - c
        out_mlu = 1 - c.mlu()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace_intscalar(self):
        type_list = [
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
            torch.double,
        ]
        for input_t in type_list:
            if input_t is torch.half:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.abs(torch.randn(20, dtype=torch.float))
                )
            else:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.abs(torch.randn(20, dtype=torch.float))
                ).to(input_t)
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_ptr = input_self_mlu.data_ptr()
            input_self_cpu -= 1
            input_self_mlu -= 1
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(
                input_self_cpu.float(), input_self_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_inplace_floatscalar(self):
        type_list = [torch.float, torch.half]
        for input_t in type_list:
            input_self_cpu = torch.normal(
                mean=20, std=torch.abs(torch.randn(20, dtype=torch.float))
            )
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_ptr = input_self_mlu.data_ptr()
            input_self_cpu -= 2.3
            input_self_mlu -= 2.3
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(
                input_self_cpu.float(), input_self_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_rsub(self):
        a = torch.randn(3, dtype=torch.float)
        b = torch.randn(3, dtype=torch.float)
        out = torch.rsub(a, b)
        out_mlu = torch.rsub(a.to("mlu"), b.to("mlu"))
        self.assertTensorsEqual(out, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_rsub_zero_dim(self):
        a = torch.tensor(3.7, dtype=torch.float)
        b = torch.tensor(2.2, dtype=torch.float)
        out = torch.rsub(a, b)
        out_mlu = torch.rsub(a.to("mlu"), b.to("mlu"))
        self.assertTensorsEqual(out, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_exception(self):
        a = torch.randn(3).bool().to("mlu")
        b = torch.randn(3).bool().to("mlu")
        ref_msg = r"^Subtraction, the \`\-\` operator, with two bool tensors is not supported\. "
        ref_msg = ref_msg + r"Use the \`\^\` or \`logical_xor\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.sub(a, b)

        a = torch.randn(3).bool().to("mlu")
        b = torch.randn(3).to("mlu")
        ref_msg = (
            r"^Subtraction, the \`\-\` operator, with a bool tensor is not supported\. "
        )
        ref_msg = ref_msg + r"If you are trying to invert a mask, use the \`\~\` or "
        ref_msg = ref_msg + r"\`logical\_not\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.sub(a, b)
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.rsub(a, b)

        a = torch.randn(3).bool().to("mlu")
        b = torch.randn(3).bool().to("mlu")
        ref_msg = r"^Subtraction, the \`\-\` operator, with two bool tensors is not supported\. "
        ref_msg = ref_msg + r"Use the \`\^\` or \`logical_xor\(\)\` operator instead\.$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.rsub(a, b)
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.rsub(a, True, 1)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_high_diemension_after_permute(self):
        for i in range(5):
            dimention_list = []
            for k in range(i + 3):  # pylint: disable=W0612
                dimention_list.append(numpy.random.randint(1, 20))
            shape = tuple(dimention_list)
            permute_size = numpy.arange(len(shape))
            # Pytorch is no longer allowed View operation returned a tensor
            # that is the same as the input base tensor.
            without_modify = True
            while without_modify:
                random.shuffle(permute_size)
                if any(permute_size != numpy.arange(len(shape))):
                    without_modify = False

            a = torch.randn(shape)
            b = torch.randn(shape)
            ouput = torch.sub(
                torch.permute(a, tuple(permute_size)),
                torch.permute(b, tuple(permute_size)),
            )
            a_mlu = torch.permute(a.mlu(), tuple(permute_size))
            b_mlu = torch.permute(b.mlu(), tuple(permute_size))
            ouput_mlu = torch.sub(a_mlu, b_mlu)
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_sub_high_diemension_after_permute_inplace(self):
        for i in range(5):
            dimention_list = []
            for k in range(i + 3):  # pylint: disable=W0612
                dimention_list.append(numpy.random.randint(1, 20))
            shape = tuple(dimention_list)
            permute_size = numpy.arange(len(shape))
            # Pytorch is no longer allowed View operation returned a tensor
            # that is the same as the input base tensor.
            without_modify = True
            while without_modify:
                random.shuffle(permute_size)
                if any(permute_size != numpy.arange(len(shape))):
                    without_modify = False

            a = torch.randn(shape)
            b = torch.randn(shape)
            a_mlu = torch.permute(a.mlu(), tuple(permute_size))
            b_mlu = torch.permute(b.mlu(), tuple(permute_size))
            a = torch.permute(a, tuple(permute_size))
            b = torch.permute(b, tuple(permute_size))

            a.sub_(b)
            a_mlu.sub_(b_mlu)
            self.assertTensorsEqual(a, a_mlu.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_sub_large(self):
        dtype_list = [(torch.half, 3e-3)]
        model = TestSubModel().float()
        for shape1, shape2 in [((5, 1024, 1024, 1024), (5, 1024, 1024, 1024))]:
            for data_type, err in dtype_list:
                input_self = torch.rand(shape1, dtype=torch.float)
                input_other = torch.rand(shape2, dtype=torch.float)
                out_cpu = model(input_self, input_other)
                out_mlu = model(
                    self.to_mlu_dtype(input_self, data_type),
                    self.to_mlu_dtype(input_other, data_type),
                )
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_sub_bfloat16(self):
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
        for dtype_err in dtype_list:
            x_left = torch.testing.make_tensor(
                (1, 3, 224, 224), dtype=dtype_err[0], device="cpu"
            )
            x_right = torch.testing.make_tensor(
                (1, 3, 224, 1), dtype=dtype_err[1], device="cpu"
            )
            x_right_cpu = torch.nn.Parameter(x_right)
            x_left_mlu = x_left.mlu()
            x_right_mlu = torch.nn.Parameter(x_right.mlu())
            out_cpu = torch.sub(x_left, x_right_cpu)
            out_mlu = torch.sub(x_left_mlu, x_right_mlu)
            # TODO(CNNLCORE-14053) : neg is not support bfloat16.
            # grad = torch.randn(out_mlu.shape)
            # grad_mlu = grad.mlu()
            # out_cpu.backward(grad)
            # out_mlu.backward(grad_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), dtype_err[2], use_MSE=True
            )
            # self.assertTensorsEqual(x_right_cpu.grad.float(), x_right_mlu.grad.cpu().float(), \
            #                         dtype_err[2], use_MSE=True)


if __name__ == "__main__":
    run_tests()
