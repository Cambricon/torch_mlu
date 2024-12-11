from __future__ import print_function

import sys
import os
import itertools
import copy
import unittest
import logging
import numpy
import random  # pylint: disable=C0411
from itertools import product
import torch

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


def to_mlu(tensor_cpu):
    return tensor_cpu.mlu()


class TestMulOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_mul(self):
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
            ((1, 3, 224, 224), (1, 3, 224, 224)),
            ((1, 3, 224, 224), (1, 3, 1, 1)),
            ((1, 1, 24, 1), (1, 1, 24, 1)),
            ((10), (1)),
            ((1, 3, 224, 1), (1, 3, 1, 224)),
            ((1, 3, 224, 1), (0, 3, 1, 224)),
            ((1, 3, 224, 224), (1, 1, 1, 1)),
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
            out_cpu = torch.mul(func(x_left), func(x_right))
            out_mlu1 = torch.mul(func(x_left_mlu), func(x_right_mlu))
            out_mlu2 = torch.multiply(func(x_left_mlu), func(x_right_mlu))
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
    def test_mul_inplace(self):
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
            ((1, 3, 224, 224), (1, 3, 224, 224)),
            ((1, 3, 224, 224), (1, 3, 1, 1)),
            ((1, 1, 24, 1), (1, 1, 24, 1)),
            ((10), (1)),
            ((1, 3, 224, 1), (1, 3, 1, 224)),
            ((1, 3, 224, 1), (0, 3, 1, 224)),
            ((1, 3, 224, 224), (1, 1, 1, 1)),
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
            x_left_cpu.mul_(func(x_right))
            x_left_mlu1.mul_(func(x_right_mlu))
            x_left_mlu2.multiply_(func(x_right_mlu))
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
    def test_mul_out(self):
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
            ((1, 3, 224, 224), (1, 3, 224, 224)),
            ((1, 3, 224, 224), (1, 3, 1, 1)),
            ((1, 1, 24, 1), (1, 1, 24, 1)),
            ((10), (1)),
            ((1, 3, 224, 1), (1, 3, 1, 224)),
            ((1, 3, 224, 1), (0, 3, 1, 224)),
            ((1, 3, 224, 224), (1, 1, 1, 1)),
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
            torch.mul(x_left_cpu, func(x_right), out=out_cpu)
            torch.mul(x_left_mlu1, func(x_right_mlu), out=out_mlu1)
            torch.multiply(x_left_mlu2, func(x_right_mlu), out=out_mlu2)
            self.assertEqual(out_cpu.dtype, out_mlu1.dtype)
            self.assertEqual(out_cpu.dtype, out_mlu2.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu1.cpu().float(), dtype_err[1], use_MSE=True
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu2.cpu().float(), dtype_err[1], use_MSE=True
            )
            # using left input as output
            torch.mul(x_left_cpu, func(x_right), out=x_left_cpu)
            torch.mul(x_left_mlu1, func(x_right_mlu), out=x_left_mlu1)
            torch.multiply(x_left_mlu2, func(x_right_mlu), out=x_left_mlu2)
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
    def test_mul_tensor_tensor_channel_last(self):
        """
        test_tensor_tensor
        """
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        param_list = [dtype_list, func_list, func_list]
        # for data_type, err in dtype_list:
        for data_type, func_x, func_y in itertools.product(*param_list):
            for shape1, shape2 in [
                ((224, 224), (1, 10, 224, 1)),
                ((1, 10, 224, 224), (1, 10, 224, 1)),
            ]:
                a = torch.rand(shape1).to(data_type)
                b = torch.rand(shape2).to(data_type)

                out_cpu = func_x(a) * func_y(b)
                out_mlu = func_x(a.to("mlu")) * func_y(b.to("mlu"))

                # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_scalar(self):
        data_types = [torch.float, torch.half]
        for shape in [(224), (2, 4, 5, 3), (24, 24)]:
            for data_type in data_types:
                b = torch.rand(shape, dtype=torch.float)
                out_cpu = 1.2 * b.sum()
                out_mlu = 1.2 * b.sum().to(data_type).mlu()
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_mul_inplace_intscalar(self):
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
                input_self_mlu = copy.deepcopy(input_self_cpu).to(input_t).mlu()
            else:
                input_self_cpu = torch.normal(
                    mean=20, std=torch.randn(20, dtype=torch.float).abs()
                ).to(input_t)
                input_self_mlu = copy.deepcopy(input_self_cpu).mlu()
            input_self_cpu *= 1
            input_self_mlu *= 1
            self.assertTensorsEqual(
                input_self_cpu.float(), input_self_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_mul_inplace_floatscalar(self):
        data_types = [torch.float, torch.half]
        for data_type in data_types:
            input_self_cpu = torch.normal(
                mean=20, std=torch.randn(20, dtype=torch.float).abs()
            )
            input_self_mlu = copy.deepcopy(input_self_cpu).to(data_type).mlu()
            input_self_cpu *= 2.3
            input_self_mlu *= 2.3
            self.assertTensorsEqual(
                input_self_cpu.float(), input_self_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_scalar_mul_dtype(self):
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
                input_self_mlu = input_self_cpu.to(type_t).mlu()
            elif type_t is torch.uint8:
                # prevent data overflow
                input_self_cpu = torch.randperm(n=63).to(type_t)
                input_self_mlu = input_self_cpu.to(type_t).mlu()
            else:
                input_self_cpu = torch.normal(
                    mean=5, std=torch.randn(5, dtype=torch.float).abs()
                ).to(type_t)
                input_self_mlu = input_self_cpu.mlu()
            out_cpu = scalar * input_self_cpu
            out_mlu = scalar * input_self_mlu
            if type_t is torch.half:
                self.assertEqual(out_mlu.dtype, torch.half)
            else:
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            if out_cpu.dtype == torch.bool:
                for val in range(out_cpu[0]):
                    self.assertEqual(out_cpu[val].item(), out_mlu.cpu()[val].item())
            else:
                self.assertTensorsEqual(
                    out_cpu.float(),
                    out_mlu.cpu().float(),
                    3e-3 if type_t is torch.half else 0.0,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_mul_out(self):
        for shape1, shape2 in [((3, 4, 2), (3, 4, 2))]:
            a = torch.randn(shape1)
            b = torch.randn(shape2)
            out_cpu = torch.randn(shape1)
            torch.mul(a, b, out=out_cpu)
            # the element number of out >= the expected of the op
            out_mlu = self.to_device(torch.randn(shape1))
            torch.mul(self.to_device(a), self.to_device(b), out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)
            # the element number of out < the expected of the op
            out_mlu = self.to_device(torch.randn((1,)))
            torch.mul(self.to_device(a), self.to_device(b), out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mul_high_diemension_after_permute(self):
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
            ouput = torch.mul(
                torch.permute(a, tuple(permute_size)),
                torch.permute(b, tuple(permute_size)),
            )

            a_mlu = torch.permute(a.mlu(), tuple(permute_size))
            b_mlu = torch.permute(b.mlu(), tuple(permute_size))
            ouput_mlu = torch.mul(a_mlu, b_mlu)
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_mul_high_diemension_after_permute_inplace(self):
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

            a.mul_(b)
            a_mlu.mul_(b_mlu)
            self.assertTensorsEqual(a, a_mlu.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_mul_large(self):
        data_types = [torch.half]
        for shape1, shape2 in [((5, 1024, 1024, 1024), (5, 1024, 1024, 1024))]:
            for data_type in data_types:
                a = torch.rand(shape1, dtype=torch.float)
                b = torch.rand(shape2, dtype=torch.float)

                out_cpu = a * b
                out_mlu = self.to_mlu_dtype(a, data_type) * self.to_mlu_dtype(
                    b, data_type
                )
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_mul_bfloat16(self):
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
            out_cpu = torch.mul(x_left, x_right)
            out_mlu = torch.mul(x_left_mlu, x_right_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), dtype_err[2], use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_mul_scalar_bfloat16(self):
        input = torch.testing.make_tensor(
            (1, 3, 224, 224), dtype=torch.bfloat16, device="cpu"
        )
        input_cpu = torch.nn.Parameter(input)
        input_mlu = torch.nn.Parameter(input.mlu())
        out_cpu = input_cpu * 3.0
        out_mlu = input_mlu * 3.0
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

    @testinfo()
    def test_mul_tensor_complex(self):
        dtype_list = [
            (torch.complex64, torch.complex64),
            (torch.complex64, torch.complex128),
            (torch.complex128, torch.complex128),
        ]
        for dtype_err in dtype_list:
            x_left = torch.testing.make_tensor(
                (1, 3, 224, 224), dtype=dtype_err[0], device="cpu"
            )
            x_right = torch.testing.make_tensor(
                (1, 3, 224, 224), dtype=dtype_err[1], device="cpu"
            )
            x_left_mlu = x_left.mlu()
            x_right_mlu = x_right.mlu()
            out_cpu = torch.mul(x_left, x_right)
            out_mlu = torch.mul(x_left_mlu, x_right_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            torch.testing.assert_close(out_cpu, out_mlu.cpu(), rtol=1.3e-6, atol=1e-5)

    @testinfo()
    def test_mul_tensor_complex_with_broadcast(self):
        shape_list = [
            ((1, 3), (2, 3)),
            ((1, 3, 224, 224), (2, 3, 224, 224)),
            ((1, 6, 24), (3, 6, 24)),
        ]
        for shape in shape_list:
            x_left = torch.testing.make_tensor(
                shape[0], dtype=torch.complex64, device="cpu"
            )
            x_right = torch.testing.make_tensor(
                shape[1], dtype=torch.complex64, device="cpu"
            )
            x_left_mlu = x_left.mlu()
            x_right_mlu = x_right.mlu()
            out_cpu = torch.mul(x_left, x_right)
            out_mlu = torch.mul(x_left_mlu, x_right_mlu)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            torch.testing.assert_close(out_cpu, out_mlu.cpu(), rtol=1.3e-6, atol=1e-5)

    @testinfo()
    def test_mul_scalar_complex(self):
        dtype_list = [torch.complex64, torch.complex128]
        for dtype_err in dtype_list:
            input_cpu = torch.testing.make_tensor(
                (1, 3, 224, 224), dtype=dtype_err, device="cpu"
            )
            input_mlu = input_cpu.mlu()
            out_cpu = input_cpu * (3.0 + 1j)
            out_mlu = input_mlu * (3.0 + 1j)
            torch.testing.assert_close(out_cpu, out_mlu.cpu(), rtol=1.3e-6, atol=1e-5)

    # @unittest.skip("not test")
    @testinfo()
    def test_mul_mixed_type(self):
        input1_types = [torch.long]
        input2_types = [torch.float, torch.int]
        shapes = [((), ()), ((), (1,)), ((2, 3, 4, 6), (3, 4, 6))]
        product_list = product(input1_types, input2_types, shapes)
        for input1_type, input2_type, shape in product_list:
            shape1, shape2 = shape
            a = torch.randint(low=1, high=10, size=shape1).to(input1_type)
            b = torch.randn(shape2, dtype=torch.float).to(input2_type)

            ouput = torch.mul(a, b)
            ouput_mlu = torch.mul(a.mlu(), b.mlu())
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 3e-3, use_MSE=True)

            ouput = torch.mul(2, b)
            ouput_mlu = torch.mul(2, b.mlu())
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 3e-3, use_MSE=True)

            ouput = torch.mul(b, 2)
            ouput_mlu = torch.mul(b.mlu(), 2)
            self.assertTensorsEqual(ouput, ouput_mlu.cpu(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    run_tests()
