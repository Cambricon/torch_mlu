from __future__ import print_function

import sys
import os
import copy

import unittest
import logging
import numpy as np

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
    skipBFloat16IfNotSupport,
)

logging.basicConfig(level=logging.DEBUG)


class TestTanhOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_tensor_scalar_contiguous(self):
        in_shape = [
            (10),
            (15, 19),
            (25, 19, 13),
            (13, 31, 16, 19),
            (14, 19, 21, 23, 21),
            (16, 17, 18, 19, 20, 21),
        ]
        for shape in in_shape:
            input_data = torch.randn(shape, dtype=torch.float)
            input_data_mlu = input_data.mlu()

            output_cpu = torch.tanh(input_data)
            output_mlu = torch.tanh(input_data_mlu)

            # test scalar
            scalar_cpu = input_data.sum()
            scalar_mlu = scalar_cpu.mlu()
            out_scalar_cpu = torch.tanh(scalar_cpu)
            out_scalar_mlu = torch.tanh(scalar_mlu)

            # test inplace operation
            input_mlu_ptr = input_data_mlu.data_ptr()
            input_data_mlu.tanh_()

            self.assertEqual(input_mlu_ptr, input_data_mlu.data_ptr())
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                out_scalar_cpu, out_scalar_mlu.cpu(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                output_cpu, input_data_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_permute(self):
        import random  # pylint: disable=C0415

        for in_shape in [
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16, 4),
            (1, 3, 16, 16, 3, 6),
            (1, 3, 16, 16, 4, 15, 8),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            size = np.arange(len(in_shape))
            random.shuffle(size)
            input_mlu = input_.to("mlu")
            input_ = torch.permute(input_, tuple(size))
            input_mlu = torch.permute(input_mlu, tuple(size))
            input_inplace_ = copy.deepcopy(input_)
            input_mlu_inplace_ = copy.deepcopy(input_mlu)
            output_cpu = torch.tanh(input_)
            output_mlu = torch.tanh(input_mlu)
            input_inplace_.tanh_()  # test inplace operation
            input_mlu_inplace_.tanh_()
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_inplace_, input_mlu_inplace_.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_tensor_scalar_channel_last(self):
        in_shape = [(13, 31, 16, 19), (14, 19, 21, 23, 21)]
        for shape in in_shape:
            input_data = torch.randn(shape, dtype=torch.float)
            input_data = self.convert_to_channel_last(input_data)
            input_data_mlu = input_data.mlu()

            output_cpu = torch.tanh(input_data)
            output_mlu = torch.tanh(input_data_mlu)

            # test inplace operation
            input_mlu_ptr = input_data_mlu.data_ptr()
            input_data_mlu.tanh_()

            self.assertEqual(input_mlu_ptr, input_data_mlu.data_ptr())
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                output_cpu, input_data_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_tensor_scalar_not_dense(self):
        in_shape = [
            (15, 19 * 2),
            (25, 19, 13 * 2),
            (13, 31, 16, 19 * 2),
            (14, 19, 21, 23, 21 * 2),
            (16, 17, 18, 19, 20, 21 * 2),
        ]
        for shape in in_shape:
            input_data = torch.empty(0)
            if len(shape) == 2:
                input_data = torch.randn(shape, dtype=torch.float)[
                    :, : int(shape[-1] / 2)
                ]
            elif len(shape) == 3:
                input_data = torch.randn(shape, dtype=torch.float)[
                    :, :, : int(shape[-1] / 2)
                ]
            elif len(shape) == 4:
                input_data = torch.randn(shape, dtype=torch.float)[
                    :, :, :, : int(shape[-1] / 2)
                ]
            elif len(shape) == 5:
                input_data = torch.randn(shape, dtype=torch.float)[
                    :, :, :, :, : int(shape[-1] / 2)
                ]
            elif len(shape) == 6:
                input_data = torch.randn(shape, dtype=torch.float)[
                    :, :, :, :, :, : int(shape[-1] / 2)
                ]
            input_data = self.convert_to_channel_last(input_data)
            input_data_mlu = input_data.mlu()

            output_cpu = torch.tanh(input_data)
            output_mlu = torch.tanh(input_data_mlu)

            # test inplace operation
            input_mlu_ptr = input_data_mlu.data_ptr()
            input_data_mlu.tanh_()

            self.assertEqual(input_mlu_ptr, input_data_mlu.data_ptr())
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                output_cpu, input_data_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_dtype(self):
        in_shape = [
            (10),
            (15, 19),
            (25, 19, 13),
            (13, 31, 16, 19),
            (14, 19, 21, 23, 21),
            (16, 17, 18, 19, 20, 21),
        ]
        # now cnnlTanh only support float and half
        type_list = [torch.float, torch.half, torch.double]
        for shape in in_shape:
            for typeId in type_list:
                input_data = torch.randn(shape, dtype=torch.float)
                input_data_cpu = input_data.to(typeId)
                input_data_mlu = input_data_cpu.mlu()

                output_cpu = torch.tanh(input_data)
                output_mlu = torch.tanh(input_data_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_tanh_backward(self):
        in_shape = [
            (50),
            (35, 46),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (16, 17, 18, 19, 20, 21),
        ]
        type_list = [torch.float, torch.half]
        for shape in in_shape:
            for typeId in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(typeId)
                x_mlu = x.mlu()

                # use float on cpu kernel
                out_cpu = x_0.tanh()
                out_mlu = x_mlu.tanh()

                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.mlu()

                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)

                self.assertTensorsEqual(
                    out_grad_cpu,
                    out_grad_mlu.cpu().float()
                    if typeId == torch.half
                    else out_grad_mlu.cpu(),
                    0.003,
                    use_MSE=True,
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("70GB")
    def test_tanh_large(self):
        in_shape = [(5, 1024, 1024, 1024)]
        type_list = [torch.half]
        for shape in in_shape:
            for typeId in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(typeId)
                x_mlu = x.mlu()

                # use float on cpu kernel
                out_cpu = x_0.tanh()
                out_mlu = x_mlu.tanh()

                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.mlu()
                out_cpu.backward(grad)
                out_grad_cpu = copy.deepcopy(x_0.grad)
                x_0.grad.zero_()
                out_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(x_0.grad)

                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_grad_cpu, out_grad_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_tanh_bfloat16(self):
        shape = [25, 19, 13]
        input_data_cpu = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
        input_data_mlu = input_data_cpu.mlu()
        grad_data_cpu = torch.randn(shape, dtype=torch.bfloat16)
        grad_data_mlu = grad_data_cpu.mlu()

        output_cpu = torch.tanh(input_data_cpu)
        output_cpu.backward(grad_data_cpu)

        input_grad_cpu = copy.deepcopy(input_data_cpu.grad)
        input_data_cpu.grad.zero_()

        output_mlu = torch.tanh(input_data_mlu)
        output_mlu.backward(grad_data_mlu)

        input_grad_mlu = copy.deepcopy(input_data_cpu.grad)

        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            input_grad_cpu, input_grad_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
