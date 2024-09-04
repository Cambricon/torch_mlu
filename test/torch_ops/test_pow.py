from __future__ import print_function

import sys
import os
import unittest
import logging
import itertools
import copy

import torch
import torch_mlu

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
logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    @staticmethod
    def generate_input_data(value):
        if isinstance(value, tuple):
            return torch.rand(value, dtype=torch.float)
        elif isinstance(value, int):
            return value
        elif isinstance(value, float):
            return value
        else:
            assert (
                False
            ), "Input type {0} not in [tuple,, int], is \
                           not support.".format(
                type(value)
            )
            return None

    # @unittest.skip("not test")
    @testinfo()
    def test_pow(self):
        shape_list = [
            (2, 3, 4),
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            1,
            3,
            5,
        ]
        exp_list = [
            0.5,
            2,
            5,
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (2, 3, 4),
            (2, 3),
            (2, 3, 4, 3, 4, 2, 1),
        ]
        data_types = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = self.generate_input_data(shape_list[i])
                exp1 = self.generate_input_data(exp_list[i])
                input1 = (
                    input1.to(data_type) if isinstance(shape_list[i], tuple) else input1
                )
                exp1 = exp1.to(data_type) if isinstance(exp_list[i], tuple) else exp1
                out_cpu = torch.pow(input1, exp1)
                out_mlu = torch.pow(
                    self.to_mlu_dtype(input1, data_type),
                    self.to_mlu_dtype(exp1, data_type),
                )
                self.assertTrue(out_cpu.dtype == out_mlu.dtype)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )
                if isinstance(exp1, int) and exp1 == 2 and data_type != torch.half:
                    input1 = torch.rand(shape_list[i], dtype=torch.float) * 10000
                    out_cpu = torch.pow(input1, exp1)
                    out_mlu = torch.pow(self.to_mlu_dtype(input1, data_type), exp1)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_out(self):
        shape_list = [
            (2, 3, 4),
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            1,
            3,
            5,
        ]
        exp_list = [
            0.5,
            2,
            5,
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (2, 3, 4),
            (2, 3),
            (2, 3, 4, 3, 4, 2, 1),
        ]
        data_types = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = self.generate_input_data(shape_list[i])
                exp1 = self.generate_input_data(exp_list[i])
                shape = shape_list[i]
                if not isinstance(shape_list[i], tuple):
                    shape = exp_list[i]
                out_cpu = self.generate_input_data(shape)
                out_mlu = self.to_mlu_dtype(out_cpu, data_type)
                torch.pow(input1, exp1, out=out_cpu)
                torch.pow(
                    self.to_mlu_dtype(input1, data_type),
                    self.to_mlu_dtype(exp1, data_type),
                    out=out_mlu,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_scalar(self):
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        for memory_format in memory_format_list:
            input_self = torch.rand(1, 3, 16, 16).to(memory_format=memory_format)
            exp1 = 3.2
            out_cpu = torch.pow(input_self, exp1)
            out_mlu = torch.pow(input_self.to("mlu"), exp1)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

            out_cpu = torch.pow(exp1, input_self)
            out_mlu = torch.pow(exp1, input_self.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

            input_mlu = copy.deepcopy(input_self)
            input_self.pow_(exp1)
            input_mlu = input_mlu.to("mlu")
            input_mlu.pow_(exp1)
            self.assertTensorsEqual(
                input_self, input_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_inplace(self):
        # input is not support scalar
        shape_list = [
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
        ]
        exp_list = [2, 5, (2, 3, 4, 3, 4, 2, 1), (2, 3, 4)]
        data_types = [(torch.float, 3e-3), (torch.half, 3e-2)]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = self.generate_input_data(shape_list[i])
                exp1 = self.generate_input_data(exp_list[i])
                input1_mlu = self.to_mlu_dtype(input1, data_type)
                exp1_mlu = self.to_mlu_dtype(exp1, data_type)
                input1_ptr = input1_mlu.data_ptr()
                input1.pow_(exp1)
                input1_mlu.pow_(exp1_mlu)
                self.assertEqual(input1_ptr, input1_mlu.data_ptr())
                self.assertTensorsEqual(
                    input1.float(), input1_mlu.cpu().float(), err, use_MSE=True
                )
                if isinstance(exp1, int) and exp1 == 2 and data_type != torch.half:
                    input1 = torch.rand(shape_list[i], dtype=torch.float) * 10000
                    input1_mlu = self.to_mlu_dtype(input1, data_type)
                    input1_ptr = input1_mlu.data_ptr()
                    input1.pow_(exp1)
                    input1_mlu.pow_(exp1)
                    self.assertEqual(input1_ptr, input1_mlu.data_ptr())
                    self.assertTensorsEqual(
                        input1.float(), input1_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_channels_last(self):
        input_shape = (2, 3, 4, 3)
        other_shapes = [(2, 3, 4, 3), (1, 1, 1, 3), 3]
        for other_shape in other_shapes:
            input_cpu = self.generate_input_data(input_shape).to(
                memory_format=torch.channels_last
            )
            if isinstance(other_shape, int):
                exp_cpu = self.generate_input_data(other_shape)
            else:
                exp_cpu = self.generate_input_data(other_shape).to(
                    memory_format=torch.channels_last
                )
            input_mlu = input_cpu.to("mlu")
            exp_mlu = self.to_mlu(exp_cpu)
            output_cpu = torch.pow(input_cpu, exp_cpu)
            output_mlu = torch.pow(input_mlu, exp_mlu)
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_memory_format_combination(self):
        input_shape = (2, 3, 4, 3)
        other_shapes = [(2, 3, 4, 3), (1, 1, 1, 3), 3]
        dtype_list = [torch.float, torch.half]
        func_list = [
            lambda x: x,
            self.convert_to_channel_last,
            lambda x: x[:, :, :, :3],
        ]
        param_list = [dtype_list, func_list, func_list]
        # for data_type, err in dtype_list:
        for data_type, func_x, func_y in itertools.product(*param_list):
            for other_shape in other_shapes:
                input_cpu = self.generate_input_data(input_shape)
                exp_cpu = self.generate_input_data(other_shape)

                input_mlu = input_cpu
                exp_mlu = exp_cpu

                if not isinstance(other_shape, int):
                    exp_cpu = func_y(exp_cpu)
                    exp_mlu = func_y(exp_mlu)

                out_cpu = torch.pow(func_x(input_cpu), exp_cpu)
                out_mlu = torch.pow(
                    func_x(self.to_mlu_dtype(input_mlu, data_type)),
                    self.to_mlu_dtype(exp_mlu, data_type),
                )

                # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float().contiguous(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_inplace_channels_last(self):
        other_shape = (2, 3, 4, 3)
        input_shape = (2, 3, 4, 3)
        exp_cpu = self.generate_input_data(other_shape).to(
            memory_format=torch.channels_last
        )
        input_cpu = self.generate_input_data(input_shape).to(
            memory_format=torch.channels_last
        )
        input_mlu = self.to_mlu(input_cpu)
        exp_mlu = exp_cpu.to("mlu")
        input_ptr = input_mlu.data_ptr()
        input_cpu.pow_(exp_cpu)
        input_mlu.pow_(exp_mlu)

        self.assertEqual(input_ptr, input_mlu.data_ptr())
        self.assertTensorsEqual(
            input_cpu.float(), input_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_not_dense(self):
        other_shape = (1, 2, 2, 5)
        input_shape = (1, 2, 2, 5)
        exp_cpu = self.generate_input_data(other_shape)
        input_cpu = self.generate_input_data(input_shape)
        input_mlu = self.to_mlu(input_cpu)
        exp_mlu = exp_cpu.to("mlu")
        output_cpu = torch.pow(input_cpu[:, :, :, :3], exp_cpu[:, :, :, :3])
        output_mlu = torch.pow(input_mlu[:, :, :, :3], exp_mlu[:, :, :, :3])
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
        )

        input_cpu[:, :, :, :3].pow_(exp_cpu[:, :, :, :3])
        input_mlu_ptr = input_mlu.data_ptr()
        input_mlu[:, :, :, :3].pow_(exp_mlu[:, :, :, :3])

        self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
        self.assertTensorsEqual(
            input_cpu.float(), input_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = self.generate_input_data(shape_list[i])
            y = self.generate_input_data(shape_list[i])
            out = self.generate_input_data(shape_list[i])
            x_mlu = copy.deepcopy(x).mlu()
            y_mlu = copy.deepcopy(y).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, y, out = (
                x.permute(permute_shape[i]),
                y.permute(permute_shape[i]),
                out.permute(permute_shape[i]),
            )
            x_mlu, y_mlu, out_mlu = (
                x_mlu.permute(permute_shape[i]),
                y_mlu.permute(permute_shape[i]),
                out_mlu.permute(permute_shape[i]),
            )
            torch.pow(x, y, out=out)
            torch.pow(x_mlu, y_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTensorsEqual(out, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_cpu_pow_mlu(self):
        a = torch.tensor(3)
        b = torch.tensor(5.0, device="mlu")
        output = a.pow(b)
        output_cpu = a.pow(b.cpu())
        self.assertTensorsEqual(output_cpu, output.cpu(), 3e-3, use_MSE=True)

        output1 = b.pow(a)
        output1_cpu = b.cpu().pow(a)
        self.assertTensorsEqual(output1_cpu, output1.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pow_backward(self):
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
                y_0 = torch.randn(shape, dtype=data_type)
                x_mlu = x_0.to("mlu")
                y_mlu = y_0.to("mlu")
                x_0.requires_grad_(True)
                y_0.requires_grad_(True)
                x_mlu.requires_grad_(True)
                y_mlu.requires_grad_(True)
                out_cpu = torch.pow(x_0, y_0)
                out_mlu = torch.pow(x_mlu, y_mlu)
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
    def test_pow_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        exp_list = [0.5]
        data_types = [(torch.half, 3e-3)]
        # Input data constraints
        # pow(x, y): -15.5 < y*log(|x|) < 15.5 should be satisfied.
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in data_types:
                input1 = self.generate_input_data(shape_list[i])
                exp1 = self.generate_input_data(exp_list[i])
                out_cpu = torch.pow(input1, exp1)
                out_mlu = torch.pow(
                    self.to_mlu_dtype(input1, data_type),
                    self.to_mlu_dtype(exp1, data_type),
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_pow_bfloat16(self):
        left = torch.testing.make_tensor((3, 4, 6), dtype=torch.bfloat16, device="cpu")
        left_cpu = torch.nn.Parameter(left)
        left_mlu = torch.nn.Parameter(left.mlu())
        # When the exp is scalar and the value is 2 or 3, the implement of cpu kernel is different from cnnl, and cnnl's precision is better. so we can not use 2 or 3 to check precision.
        # We can not use 4 too, because the backward function will calculate pow(exp - 1).
        out_cpu = torch.pow(left_cpu, 5)
        out_mlu = torch.pow(left_mlu, 5)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)
        out_mlu.backward(grad.to("mlu"))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )
        self.assertTensorsEqual(
            left_cpu.grad.float(), left_mlu.grad.cpu().float(), 3e-3, use_MSE=True
        )

        left_cpu.grad.zero_()
        left_mlu.grad.zero_()
        exponent = torch.testing.make_tensor((3, 4, 6), dtype=torch.short, device="cpu")
        out_cpu = torch.pow(left_cpu, exponent)
        out_mlu = torch.pow(left_mlu, exponent.mlu())
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)
        out_mlu.backward(grad.to("mlu"))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True, allow_inf=True
        )
        self.assertTensorsEqual(
            left_cpu.grad.float(),
            left_mlu.grad.cpu().float(),
            3e-3,
            use_MSE=True,
            allow_inf=True,
        )


if __name__ == "__main__":
    run_tests()
