# pylint: disable=W0223,R0201,C0413,C0411,C0301
from __future__ import print_function

import sys
import os
import copy
import unittest
import numpy
import random
import torch
from itertools import product
from torch import nn

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


class TestAddModel(nn.Module):
    def __init__(self):
        super(TestAddModel, self).__init__()

    def forward(self, x, y):
        z = x + y
        return z


class TestAddScaleModel(nn.Module):
    def __init__(self, scale):
        super(TestAddScaleModel, self).__init__()
        self.scale = scale

    def forward(self, x):
        y = x.add(self.scale)
        return y


class TestAddOp(TestCase):  # pylint: disable=R0904
    # @unittest.skip("not test")
    @testinfo()
    def test_add(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
            (torch.bool, 0),
        ]
        shape_list = [
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 30, 80), (2, 30, 80)),
            ((3, 20), (3, 20)),
            ((10), (10)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((2, 1, 2, 4), (2, 1, 2, 4)),
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
                shape[1], dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu = x_left.mlu()
            x_right_mlu = x_right.mlu()
            out_cpu = torch.add(func(x_left), func(x_right))
            out_mlu = torch.add(func(x_left_mlu), func(x_right_mlu))
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), dtype_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_inplace(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
            (torch.bool, 0),
        ]
        shape_list = [
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 30, 80), (2, 30, 80)),
            ((3, 20), (3, 20)),
            ((10), (10)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((2, 1, 2, 4), (2, 1, 2, 4)),
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
                shape[1], dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu = func(x_left.mlu())
            x_left_mlu_dptr = x_left_mlu.data_ptr()
            x_right_mlu = x_right.mlu()
            x_left_cpu = func(x_left)
            x_left_cpu.add_(func(x_right))
            x_left_mlu.add_(func(x_right_mlu))
            self.assertEqual(x_left.dtype, x_left_mlu.dtype)
            self.assertEqual(x_left_mlu_dptr, x_left_mlu.data_ptr())
            self.assertTensorsEqual(
                x_left_cpu.float(), x_left_mlu.cpu().float(), dtype_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_out(self):
        dtype_list = [
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int, 3e-3),
            (torch.short, 3e-3),
            (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.double, 3e-3),
            (torch.long, 3e-3),
            (torch.bool, 0),
        ]
        shape_list = [
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 30, 80), (2, 30, 80)),
            ((3, 20), (3, 20)),
            ((10), (10)),
            ((2, 1, 2, 4), (1, 2, 4)),
            ((2, 1, 2, 4), (2, 1, 2, 4)),
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
                shape[1], dtype=dtype_err[0], device="cpu"
            )
            x_left_mlu = func(x_left.mlu())
            x_right_mlu = x_right.mlu()
            x_left_cpu = func(x_left)
            # resize output
            out_cpu = torch.empty((0,), dtype=dtype_err[0])
            out_mlu = out_cpu.mlu()
            torch.add(x_left_cpu, func(x_right), out=out_cpu)
            torch.add(x_left_mlu, func(x_right_mlu), out=out_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), dtype_err[1], use_MSE=True
            )
            # using left input as output
            torch.add(x_left_cpu, func(x_right), out=x_left_cpu)
            torch.add(x_left_mlu, func(x_right_mlu), out=x_left_mlu)
            self.assertEqual(x_left_cpu.dtype, x_left_mlu.dtype)
            self.assertTensorsEqual(
                x_left_cpu.float(), x_left_mlu.cpu().float(), dtype_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_scale(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            model = TestAddScaleModel(0.5).float()
            input_self = torch.rand(1, 3, 224, 224, dtype=torch.float)
            out_cpu = model(input_self)
            out_mlu = model(self.to_mlu_dtype(input_self, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_add_scale_channel_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            model = TestAddScaleModel(0.5).float()
            input_self = torch.rand(1, 3, 224, 224).to(
                memory_format=torch.channels_last
            )
            out_cpu = model(input_self)
            out_mlu = model(self.to_mlu_dtype(input_self, data_type))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_add_scale_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            model = TestAddScaleModel(0.5).float()
            input_self = torch.rand(1, 3, 224, 224, dtype=torch.float)
            out_cpu = model(input_self[:, :, :, :112])
            out_mlu = model(self.to_mlu_dtype(input_self, data_type)[:, :, :, :112])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_add_tensor(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0.0), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            # [res] torch.add([res,] tensor1, tensor2)
            m1 = self.to_mlu_dtype(torch.randn(100, 100), data_type)
            v1 = self.to_mlu_dtype(torch.randn(100), data_type)

            ## contiguous
            res1 = torch.add(m1[4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(1)):
                res2[i] = m1[4, i] + v1[i]
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

            # non-contiguous
            res1 = torch.add(m1[:, 4], v1)
            res2 = res1.clone().zero_()
            for i in range(m1.size(0)):
                res2[i] = m1[i, 4] + v1[i]
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_inter_type(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 3e-3), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            # inter-type
            m1 = torch.randn(10, 10)
            res1 = m1 + 3
            res2 = self.to_mlu_dtype(m1, data_type) + torch.tensor(3).mlu()
            self.assertTensorsEqual(res1.cpu(), res2.cpu().float(), err, use_MSE=True)
            res1 = 3 + m1
            res2 = torch.tensor(3).mlu() + self.to_mlu_dtype(m1, data_type)
            self.assertTensorsEqual(res1.cpu(), res2.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_add_empty(self):
        dtype_list = [torch.float, torch.half, torch.double]
        for data_type in dtype_list:
            # 1d + empty
            m1 = self.to_mlu_dtype(torch.tensor([1.0], dtype=torch.float), data_type)
            m2 = self.to_mlu_dtype(torch.tensor([], dtype=torch.float), data_type)
            res = m1 + m2
            self.assertEqual(res.cpu().shape, m2.shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_add_multiply_add(self):
        # fused multiply add
        a = torch.zeros(10, dtype=torch.bool).mlu()
        res = torch.add(a, a, alpha=0)
        expected = torch.zeros(10).bool()
        for val in range(expected.shape[0]):
            self.assertTrue((res.cpu()[val].item() == expected[val].item()))

    # @unittest.skip("not test")
    @testinfo()
    def test_scalar(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            for shape in [(224), (2, 4, 5, 3), (24, 24)]:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 + b.sum()
                out_mlu = 1.2 + self.to_mlu_dtype(b.sum(), data_type)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_batchnorm_add(self):
        b = torch.tensor(100, dtype=torch.long)
        out_cpu = b + 1
        out_mlu = b.mlu() + 1
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_add_inplace_intscalar(self):
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
                    mean=20, std=torch.randn(20, dtype=torch.float).abs()
                )
            else:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float).abs()
                ).to(input_t)
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_ptr = input_self_mlu.data_ptr()
            input_self_cpu += 1
            input_self_mlu += 1
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(
                input_self_cpu.float(), input_self_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_inplace_boolscalar(self):
        input_cpu = torch.randint(100, (3, 5, 7, 9))
        input_mlu = input_cpu.to("mlu")
        input_mlu_ptr = input_mlu.data_ptr()
        input_cpu.add_(True)
        input_mlu.add_(True)
        self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
        self.assertTensorsEqual(
            input_cpu.float(), input_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_inplace_floatscalar(self):
        type_list = [torch.float, torch.half, torch.double]
        for input_t in type_list:
            if input_t is torch.half:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float).abs()
                )
            else:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float).abs()
                ).to(input_t)
            input_self_mlu = self.to_mlu_dtype(copy.deepcopy(input_self_cpu), input_t)
            input_ptr = input_self_mlu.data_ptr()
            input_self_cpu += 2.3
            input_self_mlu += 2.3
            self.assertEqual(input_ptr, input_self_mlu.data_ptr())
            self.assertTensorsEqual(
                input_self_cpu.float(), input_self_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_scalar_dtype(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
            torch.double,
        ]
        scalar_list = [3, 3.3, True]
        for scalar, type_t in product(scalar_list, type_list):
            if type_t is torch.half:
                input_self_cpu = torch.normal(
                    mean=5, std=torch.randn(5, dtype=torch.float).abs()
                )
            else:
                input_self_cpu = torch.normal(
                    mean=5, std=torch.randn(5, dtype=torch.float).abs()
                ).to(type_t)
            input_self_mlu = self.to_mlu_dtype(input_self_cpu, type_t)
            out_cpu = input_self_cpu + scalar
            out_mlu = input_self_mlu + scalar
            if out_cpu.dtype == torch.bool:
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                for val in range(out_cpu[0]):
                    self.assertEqual(out_cpu[val].item(), out_mlu.cpu()[val].item())
            elif out_mlu.dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
                )
            else:
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_scalar_add_dtype(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
            torch.half,
            torch.double,
        ]
        scalar_list = [3, 3.3, True]
        for scalar, type_t in product(scalar_list, type_list):
            if type_t is torch.half:
                input_self_cpu = torch.normal(
                    mean=5, std=torch.randn(5, dtype=torch.float).abs()
                )
            else:
                input_self_cpu = torch.normal(
                    mean=5, std=torch.randn(5, dtype=torch.float).abs()
                ).to(type_t)
            input_self_mlu = self.to_mlu_dtype(input_self_cpu, type_t)
            out_cpu = scalar + input_self_cpu
            out_mlu = scalar + input_self_mlu
            if out_cpu.dtype == torch.bool:
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                for val in range(out_cpu[0]):
                    self.assertEqual(out_cpu[val].item(), out_mlu.cpu()[val].item())
            elif out_mlu.dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
                )
            else:
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_value(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0.0), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            # [res] torch.add([res,] tensor, value)
            m1 = self.to_mlu_dtype(torch.randn(10, 10), data_type)

            # contiguous
            res1 = m1.clone()
            res1[1].add_(2)
            res2 = m1.clone()
            for i in range(m1.size(1)):
                res2[1, i] = res2[1, i] + 2
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

            # non-contiguous
            m1 = self.to_mlu_dtype(torch.randn(10, 10), data_type)
            res1 = m1.clone()
            res1[:, 3].add_(2)
            res2 = m1.clone()
            for i in range(m1.size(0)):
                res2[i, 3] = res2[i, 3] + 2
            self.assertTensorsEqual(
                res1.cpu().float(), res2.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_channels_last(self):
        shapes_list = [
            ((64, 3, 7, 7), (7, 7)),
            ((14, 7, 7, 7), (7)),
            ((3, 4, 5), (2, 3, 4, 5)),
            ((3, 3, 3), (3, 3, 3, 3)),
            ((5, 5, 5, 5), (5, 5, 5, 5)),
            ((5, 5, 5, 5), (1, 5, 5, 5, 5)),
        ]
        for shape1, shape2 in shapes_list:
            input = torch.randn(shape1, dtype=torch.float)
            other = torch.randn(shape2, dtype=torch.float)
            input_mlu = self.convert_to_channel_last(input.to("mlu"))
            other_mlu = self.convert_to_channel_last(other.to("mlu"))
            input_cl = self.convert_to_channel_last(input)
            other_cl = self.convert_to_channel_last(other)

            # channels_last
            output_cpu = input + other
            output_mlu = input_mlu + other_mlu
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.00, use_MSE=True)

            # channels_last and inplace
            if input_cl.dim() >= other_cl.dim():
                input_cl.add_(other_cl)
                input_mlu_ptr = input_mlu.data_ptr()
                input_mlu.add_(other_mlu)
                self.assertTensorsEqual(input_cl, input_mlu.cpu(), 0.00, use_MSE=True)
                self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_add_not_dense(self):
        shapes_list = [((64, 3, 7, 7), (7, 7))]
        for shape1, shape2 in shapes_list:
            input = torch.randn(shape1, dtype=torch.float)
            other = torch.randn(shape2, dtype=torch.float)
            input_mlu = input.to("mlu")
            other_mlu = other.to("mlu")

            output_cpu = input[:, :, :, :5] + other[:, :5]
            output_mlu = input_mlu[:, :, :, :5] + other_mlu[:, :5]
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.00, use_MSE=True)

            if input.dim() >= other.dim():
                input[:, :, :, :5].add_(other[:, :5])
                input_mlu_ptr = input_mlu.data_ptr()
                input_mlu[:, :, :, :5].add_(other_mlu[:, :5])
                self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)
                self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_add_exception(self):
        a = torch.randn(3).to("mlu")
        b = torch.randn(3).to("mlu")
        ref_msg = "Boolean alpha only supported for Boolean results"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.add(a, b, alpha=True)

        a = torch.randn(3).int().to("mlu")
        b = torch.randn(3).int().to("mlu")
        ref_msg = "For integral input tensors, argument alpha must not be a floating point number."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.add(a, b, alpha=2.1)

    # @unittest.skip("not test")
    @testinfo()
    def test_add_high_diemension_after_permute(self):
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
            ouput_ture = torch.add(
                torch.permute(a, tuple(permute_size)),
                torch.permute(b, tuple(permute_size)),
            )

            a_mlu = torch.permute(a.mlu(), tuple(permute_size))
            b_mlu = torch.permute(b.mlu(), tuple(permute_size))
            ouput_ture_mlu = torch.add(a_mlu, b_mlu)
            self.assertTensorsEqual(
                ouput_ture, ouput_ture_mlu.cpu(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_add_high_diemension_after_permute_inplace(self):
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

            a.add_(b)
            a_mlu.add_(b_mlu)
            self.assertTensorsEqual(a, a_mlu.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_add_large(self):
        model = TestAddModel().float()
        dtype_list = [(torch.half, 3e-3)]
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
    def test_add_bfloat16(self):
        dtype_list = [
            (torch.float, torch.bfloat16, 3e-3),
            (torch.half, torch.bfloat16, 3e-3),
            (torch.int, torch.bfloat16, 3e-3),
            (torch.short, torch.bfloat16, 3e-3),
            (torch.int8, torch.bfloat16, 3e-3),
            (torch.uint8, torch.bfloat16, 3e-3),
            (torch.double, torch.bfloat16, 3e-3),
            (torch.long, torch.bfloat16, 3e-3),
            (torch.bool, torch.bfloat16, 3e-3),
            (torch.bfloat16, torch.bfloat16, 0),
        ]
        for dtype_err in dtype_list:
            x_left = torch.testing.make_tensor(
                (1, 3, 224, 224), dtype=dtype_err[0], device="cpu"
            )
            x_right = torch.testing.make_tensor(
                (1, 3, 224, 1), dtype=dtype_err[1], device="cpu"
            )
            x_left_mlu = x_left.mlu()
            x_right_mlu = x_right.mlu()
            out_cpu = torch.add(x_left, x_right)
            out_mlu = torch.add(x_left_mlu, x_right_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), dtype_err[2], use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_add_scalar_bfloat16(self):
        input = torch.testing.make_tensor(
            (1, 3, 224, 224), dtype=torch.bfloat16, device="cpu"
        )
        input_cpu = torch.nn.Parameter(input)
        input_mlu = torch.nn.Parameter(input.mlu())
        out_cpu = input_cpu + 2
        out_mlu = input_mlu + 2
        grad = torch.randn(out_cpu.shape)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertEqual(out_cpu.dtype, out_mlu.dtype)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
