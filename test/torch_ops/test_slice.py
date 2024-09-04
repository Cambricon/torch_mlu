from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import copy

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TEST_BFLOAT16,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestSliceOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_slice(self):
        in_shape = (2, 3, 24, 30)
        in_shape1 = (2, 3, 33)
        in_shape2 = (2, 24)
        input_dtypes = [torch.float, torch.half]
        channel_first = [False, True]
        for data_type in input_dtypes:
            for channel in channel_first:
                input_t = torch.rand(in_shape, dtype=torch.float)
                input1 = torch.rand(in_shape1, dtype=torch.float)
                input2 = torch.rand(in_shape2, dtype=torch.float)
                output_cpu = input_t[:, 1:, 2:-1:3, 10:20][0:1:1, 1:2:1, 3:-1:2, 2:5:2]
                input_mlu = self.to_mlu_dtype(input_t, data_type)
                if channel is False:
                    input_t = self.convert_to_channel_last(input_t)
                    input_mlu = self.convert_to_channel_last(input_mlu)
                output_mlu = input_mlu[:, 1:, 2:-1:3, 10:20][
                    0:1:1, 1:2:1, 3:-1:2, 2:5:2
                ]
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )
                output_cpu1 = input1[:, 1:, :]
                input1_mlu = self.to_mlu_dtype(input1, data_type)
                output_mlu1 = input1_mlu[:, 1:, :]
                self.assertTensorsEqual(
                    output_cpu1, output_mlu1.cpu().float(), 0.003, use_MSE=True
                )
                output_cpu1 = input2[1:, 10:]
                input2_mlu = self.to_mlu_dtype(input2, data_type)
                output_mlu1 = input2_mlu[1:, 10:]
                self.assertTensorsEqual(
                    output_cpu1, output_mlu1.cpu().float(), 0.003, use_MSE=True
                )
                output_cpu = input_t[:, :, :, -2:][0:-1:1, :, -20:-2:2, ...]
                output_mlu = input_mlu[:, :, :, -2:][0:-1:1, :, -20:-2:2, ...]
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_slice_optimization(self):
        in_shape0 = (4, 4, 4, 4)
        in_shape1 = (8, 6, 3)
        in_shape2 = 24
        in_shape3 = (10, 10)
        in_shape4 = 1000
        input0 = torch.rand(in_shape0, dtype=torch.float)
        input0_mlu = input0.to("mlu")
        input1 = torch.rand(in_shape1, dtype=torch.float)
        input1_mlu = input1.to("mlu")
        input2 = torch.rand(in_shape2, dtype=torch.float)
        input2_mlu = input2.to("mlu")
        input3 = torch.rand(in_shape3, dtype=torch.float)
        input3_mlu = input3.to("mlu")
        input4 = torch.rand(in_shape4, dtype=torch.float)
        input4_mlu = input4.to("mlu")
        output_cpu = input0[:, :, 1:4, :].contiguous()
        output_mlu = input0_mlu[:, :, 1:4, :].contiguous()
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.00, use_MSE=True
        )
        output_cpu = input1[:, 3:6, :].contiguous()
        output_mlu = input1_mlu[:, 3:6, :].contiguous()
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.00, use_MSE=True
        )
        output_cpu = input2.as_strided((3,), (3,)).contiguous()
        output_mlu = input2_mlu.as_strided((3,), (3,)).contiguous()
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.00, use_MSE=True
        )
        output_cpu = input3[3:8, :].contiguous()
        output_mlu = input3_mlu[3:8, :].contiguous()
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.00, use_MSE=True
        )
        output_cpu = input0.as_strided((4, 2, 4, 4), (64, 32, 4, 1)).contiguous()
        output_mlu = input0_mlu.as_strided((4, 2, 4, 4), (64, 32, 4, 1)).contiguous()
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.00, use_MSE=True
        )
        output_cpu = input4.as_strided((5, 6, 1), (64, 1, 666)).contiguous()
        output_mlu = input4_mlu.as_strided((5, 6, 1), (64, 1, 666)).contiguous()
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 0.00, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_slice_backward(self):
        x = torch.randn((30, 2), requires_grad=True)
        x_mlu = self.to_device(x)
        z = x[1:12]
        z_mlu = x_mlu[1:12]
        grad = torch.randn(11, 2)
        grad_mlu = self.to_device(grad)
        z.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        x.grad.zero_()
        z_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(z, z_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

        x = torch.randn((5, 2), requires_grad=True)
        x_mlu = self.to_device(x)
        z = x[1:]
        z_mlu = x_mlu[1:]
        grad = torch.randn(4, 2)
        grad_mlu = self.to_device(grad)
        z.backward(grad)
        out_grad = copy.deepcopy(x.grad)
        x.grad.zero_()
        z_mlu.backward(grad_mlu)
        out_grad_mlu = x.grad
        self.assertTensorsEqual(z, z_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(out_grad, out_grad_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_slice_exception(self):
        a = torch.tensor(5, dtype=torch.float).to("mlu")
        ref_msg = r"Dimension specified as 0 but tensor has no dimensions"
        with self.assertRaisesRegex(IndexError, ref_msg):
            b = a[0:1:1]

        a = torch.randn((2, 3, 4), dtype=torch.float).to("mlu")
        ref_msg = r"step must be greater than zero"
        with self.assertRaisesRegex(ValueError, ref_msg):
            b = a[0:1:-1, :]

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_slice_bfloat16(self):
        input_t = torch.rand((2, 3, 24, 30), dtype=torch.bfloat16)
        input_cpu = torch.nn.Parameter(input_t)
        input_mlu = torch.nn.Parameter(input_t.mlu())
        output_cpu = input_cpu[:, 1:, 2:-1:3, 10:20][0:1:1, 1:2:1, 3:-1:2, 2:5:2]
        output_mlu = input_mlu[:, 1:, 2:-1:3, 10:20][0:1:1, 1:2:1, 3:-1:2, 2:5:2]
        grad = torch.randn(output_cpu.shape).to(torch.bfloat16)
        output_cpu.backward(grad)
        output_mlu.backward(grad.mlu())
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(
            input_cpu.grad, input_mlu.grad.cpu(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_slice_large(self):
        input_t = torch.rand((5, 1024, 1024, 1024))
        input_cpu = torch.nn.Parameter(input_t)
        input_mlu = torch.nn.Parameter(input_t.mlu())
        output_cpu = input_cpu[:, 1:, 2:-1:3, 10:20][0:1:1, 1:2:1, 3:-1:2, 2:5:2]
        output_mlu = input_mlu[:, 1:, 2:-1:3, 10:20][0:1:1, 1:2:1, 3:-1:2, 2:5:2]
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
