from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import random
import numpy as np
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
    skipBFloat16IfNotSupport,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestSigmoidOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid(self):
        # mlu device support torch.half, while cpu not
        type_list = [torch.float]
        for Type in type_list:
            # test_dim_0
            x_0 = torch.tensor(8.0, dtype=Type)

            out_cpu = torch.sigmoid(x_0)
            out_mlu = torch.sigmoid(copy.deepcopy(x_0).to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003)

            for in_shape in [
                (1),
                (2, 3),
                (8, 224, 224),
                (1, 1, 1, 1),
                (1, 3, 16, 16),
                (1, 3, 16, 16, 3),
            ]:
                input_mlu = torch.randn(in_shape, dtype=Type)
                input_cpu = copy.deepcopy(input_mlu)
                input_mlu_raw = copy.deepcopy(input_mlu)

                output_cpu = torch.sigmoid(input_cpu)
                output_mlu = torch.sigmoid(input_mlu.to("mlu"))

                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
                self.assertTensorsEqual(input_mlu_raw, input_mlu, 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_permute(self):
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
            output_cpu = torch.sigmoid(input_)
            output_mlu = torch.sigmoid(input_mlu)
            input_inplace_.sigmoid_()  # test inplace operation
            input_mlu_inplace_.sigmoid_()
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_inplace_, input_mlu_inplace_.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_channel_last(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_mlu = torch.randn(in_shape).to(memory_format=torch.channels_last)
            input_cpu = copy.deepcopy(input_mlu)
            input_mlu_raw = copy.deepcopy(input_mlu)

            output_cpu = torch.sigmoid(input_cpu)
            output_mlu = torch.sigmoid(input_mlu.to("mlu"))

            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_mlu_raw, input_mlu, 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_not_dense(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_mlu = torch.randn(in_shape)
            input_cpu = copy.deepcopy(input_mlu)
            input_mlu_raw = copy.deepcopy(input_mlu)
            output_cpu = torch.sigmoid(input_cpu[:, :, :, :2])
            output_mlu = torch.sigmoid(input_mlu.to("mlu")[:, :, :, :2])

            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_mlu_raw, input_mlu, 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_inplace(self):
        # mlu device support torch.half, while cpu not
        type_list = [torch.float]
        for Type in type_list:
            # test_dim_0
            x_0 = torch.tensor(8.0, dtype=Type)

            out_cpu = x_0
            out_mlu = copy.deepcopy(x_0).to("mlu")
            out_cpu.sigmoid_()
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu.sigmoid_()
            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003)

            for in_shape in [
                (1),
                (2, 3),
                (8, 224, 224),
                (1, 1, 1, 1),
                (1, 3, 16, 16),
                (1, 3, 16, 16, 3),
            ]:
                input_cpu = torch.randn(in_shape, dtype=Type)
                input_mlu = copy.deepcopy(input_cpu).to("mlu")

                input_cpu.sigmoid_()
                input_mlu_ptr = input_mlu.data_ptr()
                input_mlu.sigmoid_()

                self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
                self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_inplace_channel_last(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_cpu = torch.randn(in_shape).to(memory_format=torch.channels_last)
            input_mlu = copy.deepcopy(input_cpu).to("mlu")

            input_cpu.sigmoid_()
            input_mlu_ptr = input_mlu.data_ptr()
            input_mlu.sigmoid_()

            self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_inplace_not_dense(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_cpu = torch.randn(in_shape)
            input_mlu = copy.deepcopy(input_cpu).to("mlu")

            input_cpu[:, :, :, :2].sigmoid_()
            input_mlu_ptr = input_mlu.data_ptr()
            input_mlu[:, :, :, :2].sigmoid_()

            self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_backward(self):
        for in_shape in [
            (1),
            (2, 3),
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            x = torch.randn(in_shape, dtype=torch.float, requires_grad=True)
            x_mlu = x.to("mlu")

            # use float on cpu kernel
            out_cpu = x.sigmoid()
            out_mlu = x_mlu.sigmoid()

            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.to("mlu")

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()

            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu().float(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_backward_broadcast(self):
        grad_output = torch.randn(2, 15, 10)
        output = torch.randn(1, 15, 10)
        grad_input_cpu = torch.ops.aten.sigmoid_backward(grad_output, output)
        grad_input_mlu = torch.ops.aten.sigmoid_backward(
            grad_output.mlu(), output.mlu()
        )
        self.assertTensorsEqual(grad_input_cpu, grad_input_mlu.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_sigmoid_nan_inf(self):
        x_nan = torch.tensor([float("nan")], dtype=torch.float, requires_grad=True)
        x_nan_mlu = x_nan.mlu()
        out_nan = torch.sigmoid(x_nan)
        out_nan_mlu = torch.sigmoid(x_nan_mlu)
        self.assertEqual(x_nan, out_nan_mlu.cpu())
        grad = torch.tensor([1.0], dtype=torch.float)
        out_nan.backward(grad)
        grad_cpu = copy.deepcopy(x_nan.grad)
        x_nan.grad.zero_()
        out_nan_mlu.backward(grad.mlu())
        grad_mlu = x_nan.grad
        self.assertEqual(grad_cpu, grad_mlu.cpu())

        x_inf = torch.tensor([float("inf")], dtype=torch.float)
        x_inf_mlu = x_inf.mlu()
        out_cpu = torch.sigmoid(x_inf)
        out_inf_mlu = torch.sigmoid(x_inf_mlu)
        self.assertEqual(out_cpu, out_inf_mlu.cpu())

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("70GB")
    def test_sigmoid_large(self):
        in_shape = [(5, 1024, 1024, 1024)]
        type_list = [torch.half]
        for shape in in_shape:
            for typeId in type_list:
                x_0 = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x = x_0.to(typeId)
                x_mlu = x.to("mlu")

                # use float on cpu kernel
                out_cpu = x_0.sigmoid()
                out_mlu = x_mlu.sigmoid()

                grad = torch.randn(out_cpu.shape)
                grad_mlu = grad.to("mlu")

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
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_sigmoid_bfloat16(self):
        for in_shape in [(2, 3, 16, 16)]:
            input_mlu = torch.randn(in_shape, dtype=torch.bfloat16)
            input_cpu = copy.deepcopy(input_mlu)
            input_mlu_raw = copy.deepcopy(input_mlu)

            output_cpu = torch.sigmoid(input_cpu)
            output_mlu = torch.sigmoid(input_mlu.to("mlu"))

            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
            self.assertTensorsEqual(input_mlu_raw, input_mlu, 0.003)


if __name__ == "__main__":
    run_tests()
