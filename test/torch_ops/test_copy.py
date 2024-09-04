from __future__ import print_function

import sys
import os
import unittest
import logging
import itertools

import torch
import numpy as np
from torch import nn
from itertools import product

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestCopyOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_copy_D2D(self):
        a = self.to_mlu(torch.randn(3)) + 1
        a_data_ptr = a.data_ptr()
        b = self.to_mlu(torch.randn(3))
        a.copy_(b)
        self.assertEqual(a_data_ptr, a.data_ptr())
        self.assertTensorsEqual(a.cpu(), b.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_H2D(self):
        dtype_list = [
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.long,
            torch.int8,
            torch.bool,
            torch.uint8,
        ]
        for dtype in dtype_list:
            a = self.to_mlu(torch.randn(3, dtype=torch.float)) + 1
            a_data_ptr = a.data_ptr()
            b = torch.randn(3).to(dtype)
            a.copy_(b)
            self.assertEqual(b.dtype, dtype)
            self.assertEqual(a_data_ptr, a.data_ptr())
            self.assertEqual(a.dtype, torch.float)
            self.assertTensorsEqual(a.cpu(), b, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_H2D_long_double(self):
        shape_list = [(32, 13, 50, 9), (5, 7, 8, 5), (5, 10)]
        dtype_list = [torch.long, torch.double]
        for shape in shape_list:
            for dtype in dtype_list:
                input_cpu = torch.randn(shape).to(dtype) + 1
                result_cpu = torch.randn(shape).to(dtype) + 1
                input_mlu = input_cpu.to("mlu")
                result_mlu = result_cpu.to("mlu")
                # Generate random slice index
                index = 2 if len(shape) > 2 else 1
                start = np.random.randint(0, shape[index] - 3)
                end = np.random.randint(shape[index] - 2, shape[index])
                step = np.random.randint(1, end - start)
                if len(shape) > 2:
                    result_cpu[:, :, start:end:step, :] = input_cpu[
                        :, :, start:end:step, :
                    ]
                    result_mlu[:, :, start:end:step, :] = input_mlu[
                        :, :, start:end:step, :
                    ]
                else:
                    result_cpu[:, start:end:step] = input_cpu[:, start:end:step]
                    result_mlu[:, start:end:step] = input_mlu[:, start:end:step]
                self.assertEqual(result_cpu.dtype, result_mlu.dtype)
                self.assertTensorsEqual(
                    result_cpu.float(), result_mlu.cpu().float(), 0.0, use_MSE=True
                )
                # without input slice.
                input_shape = list(shape)
                index_size = (end - start + step - 1) // step
                input_shape[index] = index_size
                input_cpu = torch.randn(input_shape).to(dtype) + 1
                result_cpu = torch.randn(shape).to(dtype) + 1
                input_mlu = input_cpu.to("mlu")
                result_mlu = result_cpu.to("mlu")
                if len(shape) > 2:
                    result_cpu[:, :, start:end:step, :] = input_cpu
                    result_mlu[:, :, start:end:step, :] = input_mlu
                else:
                    result_cpu[:, start:end:step] = input_cpu
                    result_mlu[:, start:end:step] = input_mlu
                self.assertEqual(result_cpu.dtype, result_mlu.dtype)
                self.assertTensorsEqual(
                    result_cpu.float(), result_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_complex(self):
        dtype_list = [torch.complex32, torch.complex64, torch.complex128]
        for dtype in dtype_list:
            a = torch.randn(3, dtype=dtype)
            b = torch.randn(3, dtype=dtype)
            am = a.to("mlu", non_blocking=True)
            bm = b.to("mlu", non_blocking=True)
            bm.copy_(am)
            self.assertEqual(a.dtype, bm.dtype)
            self.assertTensorsEqual(
                torch.view_as_real(a).float(), torch.view_as_real(bm.cpu()).float(), 0.0
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_cast_H2D(self):
        a = self.to_mlu(torch.randn(3)) + 1
        a_data_ptr = a.data_ptr()
        b = torch.randn(3).int()
        a.copy_(b)
        self.assertEqual(a_data_ptr, a.data_ptr())
        self.assertTensorsEqual(a.cpu(), b.to(a.dtype), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_cast_D2H(self):
        a = self.to_device(torch.randn(3))
        b0 = torch.randint(10, (3,))
        b_data_ptr = b0.data_ptr()
        b1 = b0.clone()
        b0.copy_(a)
        b1.copy_(a.cpu())
        self.assertEqual(b_data_ptr, b0.data_ptr())
        self.assertTensorsEqual(b0, b1, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_cast_D2D(self):
        a = self.to_mlu(torch.randn(3)) + 1
        a_data_ptr = a.data_ptr()
        b = self.to_mlu(torch.randn(3)).int()
        a.copy_(b)
        self.assertEqual(a_data_ptr, a.data_ptr())
        self.assertTensorsEqual(a.cpu(), b.to(a.dtype).cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_before_in_module(self):
        cm = nn.Conv2d(3, 64, 7, bias=False).mlu()
        local_name_params = itertools.chain(cm._parameters.items(), cm._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}
        input_param = torch.randn(cm.weight.shape)
        for _, param in local_state.items():
            if param.shape == (64, 3, 7, 7):
                param.copy_(input_param)
                self.assertTensorsEqual(param.cpu(), input_param, 0.0, use_MSE=True)
        self.assertTensorsEqual(cm.weight.cpu(), input_param, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_to_device(self):
        shape = (32, 3, 24, 24)
        for d in range(torch.mlu.device_count()):
            x = torch.randn(shape, dtype=torch.float)
            device = torch.device("mlu:" + str(d))
            x_mlu = x.to(device)
            self.assertEqual(device, x_mlu.device)
            out_mlu = torch.abs(x_mlu)
            self.assertEqual(device, out_mlu.device)
            self.assertTensorsEqual(torch.abs(x), out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_to_non_blocking(self):
        shape = (32, 3, 24, 24)
        for d in range(torch.mlu.device_count()):
            x = torch.randn(shape, dtype=torch.float)
            device = torch.device("mlu:" + str(d))
            x_mlu = x.to(device, non_blocking=True)
            self.assertEqual(device, x_mlu.device)
            out_mlu = torch.abs(x_mlu)
            self.assertEqual(device, out_mlu.device)
            self.assertTensorsEqual(torch.abs(x), out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_device_to_device(self):
        shape = (64, 3, 1080, 1920)
        x_float = torch.randn(shape, dtype=torch.float)
        x_int64 = torch.randint(low=-100, high=100, size=shape, dtype=torch.int64)
        for x in [x_float, x_int64]:
            mlu0 = torch.device("mlu:0")
            x_mlu = x.to(mlu0)
            self.assertEqual(x_mlu.device, mlu0)
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

            if torch.mlu.device_count() < 2:
                return
            mlu1 = torch.device("mlu:1")
            x_device = x_mlu.to(mlu1)
            self.assertEqual(x_device.device, mlu1)
            self.assertTensorsEqual(x, x_device.cpu(), 0.0, use_MSE=True)

            y_mlu1 = torch.randn(shape, dtype=torch.float).to(mlu1)
            y_mlu1[:, :, 7:996, :].copy_(x_mlu[:, :, 7:996, :])
            self.assertTensorsEqual(
                y_mlu1[:, :, 7:996, :].cpu(),
                x_mlu[:, :, 7:996, :].cpu(),
                0.0,
                use_MSE=True,
            )

            z_mlu1 = x_mlu[:, :, 7:996, :].to(mlu1)
            self.assertTensorsEqual(
                z_mlu1.cpu(), x_mlu[:, :, 7:996, :].cpu(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_stride_H2D(self):
        N = 1
        C = 3
        H = 224
        W = 224
        a = torch.randn(N, C, H, W)
        a_mlu = a.to("mlu")
        self.assertEqual(a.stride(), a_mlu.stride())

        b = a.transpose(0, 1)
        b_mlu = b.to("mlu")
        self.assertEqual(b.stride(), b_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_stride_D2D(self):
        N = 2
        C = 3
        H = 224
        W = 224
        a = torch.randn(N, C, H, W)
        a_mlu = a.to("mlu")
        self.assertEqual(a.stride(), a_mlu.stride())

        b_mlu = a_mlu.transpose(0, 1)
        c_mlu = a.transpose(0, 1).to("mlu")

        b_mlu.copy_(c_mlu)
        self.assertEqual(b_mlu.stride(), c_mlu.stride())
        self.assertTensorsEqual(b_mlu.cpu(), c_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_not_contiguous_D2D(self):
        N = 1
        C = 3
        H = 224
        W = 224
        a = torch.randn(N, H, C, W)
        a_mlu = a.to("mlu")
        self.assertEqual(a.stride(), a_mlu.stride())
        b = a.transpose(1, 2)
        b_mlu = a_mlu.transpose(1, 2)
        self.assertEqual(b.stride(), b_mlu.stride())
        c = torch.randn(W, H, C, N)
        c_mlu = c.to("mlu")
        d = c.permute(3, 2, 1, 0)
        d_mlu = c_mlu.permute(3, 2, 1, 0)
        self.assertEqual(d.stride(), d_mlu.stride())
        b_mlu.copy_(d_mlu)
        self.assertTensorsEqual(b_mlu.cpu(), d_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_not_contiguous_H2D(self):
        N = 1
        C = 3
        H = 224
        W = 224
        a = torch.randn(N, H, C, W)
        a_mlu = a.to("mlu")
        self.assertEqual(a.stride(), a_mlu.stride())
        b = a.transpose(1, 2)
        b_mlu = a_mlu.transpose(1, 2)
        self.assertEqual(b.stride(), b_mlu.stride())
        c = torch.randn(W, H, C, N)
        d = c.permute(3, 2, 1, 0)
        b_mlu.copy_(d)
        self.assertEqual(b.stride(), b_mlu.stride())
        self.assertTensorsEqual(b_mlu.cpu(), d.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_not_contiguous_D2H(self):
        shape = (1, 3, 24, 24)
        dtype_list = [
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.double,
            torch.long,
            torch.bool,
            torch.complex64,
            torch.complex128,
            torch.complex32,
        ]
        if torch.mlu.is_bf16_supported():
            dtype_list.append(torch.bfloat16)
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype, func in product(dtype_list, func_list):
            err = 0.0
            # implicit cast when 64bit copy.
            if dtype in (torch.double, torch.long):
                err = 0.003
            a = torch.testing.make_tensor(shape, dtype=dtype, device="cpu")
            if a.is_complex():
                a_mlu = func(a.mlu())
            else:
                a_mlu = func(a.mlu().fill_(0))
            a_cpu = func(a)
            base_size = a_cpu.size()
            base_stride = a_cpu.stride()
            base_type = a_cpu.dtype
            base_data_ptr = a_cpu.data_ptr()
            # H2D
            a_mlu.copy_(a_cpu)
            self.assertEqual(base_size, a_mlu.size())
            self.assertEqual(base_stride, a_mlu.stride())
            self.assertEqual(base_type, a_mlu.dtype)
            self.assertTensorsEqual(
                a_cpu.float(), a_mlu.cpu().float(), err, use_MSE=True
            )
            # D2H
            a_cpu.fill_(0).copy_(a_mlu)
            self.assertEqual(base_size, a_cpu.size())
            self.assertEqual(base_stride, a_cpu.stride())
            self.assertEqual(base_type, a_cpu.dtype)
            self.assertEqual(base_data_ptr, a_cpu.data_ptr())
            self.assertTensorsEqual(func(a).float(), a_cpu.float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_expand(self):
        shape_list = [((16, 3, 22, 22), (22, 22)), ((3, 4, 8), (1, 8))]
        for shape1, shape2 in shape_list:
            x = torch.randn(shape2)
            y = torch.randn(shape1)
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            y.copy_(x)
            y_mlu.copy_(x_mlu)
            self.assertTensorsEqual(y_mlu.cpu(), y, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_cpu_to_device(self):
        shape_device = (16, 3, 224, 224)
        shape_cpu = (1, 16, 24, 24)
        x_mlu = torch.randn(shape_device).mlu()
        y_cpu = torch.ones(shape_cpu)
        x_mlu.data = y_cpu
        self.assertTensorsEqual(x_mlu.data, y_cpu, 0.0, use_MSE=True)
        self.assertEqual(x_mlu.device, y_cpu.device)

    # @unittest.skip("not test")
    @testinfo()
    def test_copy_device_to_cpu(self):
        shape_device = (16, 3, 224, 224)
        shape_cpu = (1, 16, 24, 24)
        x_mlu = torch.randn(shape_device).mlu()
        y_cpu = torch.ones(shape_cpu)
        y_cpu = x_mlu
        self.assertTensorsEqual(x_mlu.cpu(), y_cpu.cpu(), 0.0, use_MSE=True)
        self.assertEqual(y_cpu.device, x_mlu.device)

    ## @unittest.skip("not test")
    @testinfo()
    def test_copy_channel_last_D2D(self):
        shape_list = [(32, 13, 4, 50), (2, 3, 4, 5)]
        for shape in shape_list:
            a = torch.randn(shape).to(memory_format=torch.channels_last)
            b = a.to("mlu")
            self.assertEqual(a.dtype, b.dtype)
            self.assertTensorsEqual(a.float(), b.cpu().float(), 0.0, use_MSE=True)

            input_cpu = torch.randn(shape)
            input_mlu = input_cpu.to("mlu").to(memory_format=torch.channels_last)
            b.copy_(input_mlu)
            self.assertEqual(b.dtype, input_cpu.dtype)
            self.assertTensorsEqual(
                input_cpu.float(), b.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_not_default_stride_to_channel_last_D2D(self):
        shape_list = [(32, 13, 4, 50), (2, 3, 4, 5)]
        for shape in shape_list:
            a = torch.randn(shape).to(memory_format=torch.channels_last)
            input_cpu = a[0].unsqueeze(0)

            output_mlu = input_cpu.to("mlu").to(memory_format=torch.channels_last)
            output_cpu = input_cpu.to(memory_format=torch.channels_last)
            self.assertEqual(output_cpu.stride(), output_mlu.stride())
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_non_overlapping_and_dense_H2D(self):
        shape_order_list = [
            [(3, 3), (1, 0)],
            [(5, 7), (1, 0)],
            [(3, 3, 3), (1, 2, 0)],
            [(3, 5, 7), (1, 0, 2)],
            [(32, 13, 4, 50), (2, 3, 0, 1)],
            [(3, 3, 3, 3), (1, 3, 2, 0)],
            [(3, 3, 3, 3, 3), (4, 1, 2, 0, 3)],
            [(2, 3, 5, 7, 11), (3, 1, 4, 0, 2)],
        ]

        for shape, order in shape_order_list:
            a = torch.randn(shape, dtype=torch.float).permute(order)
            b = a.to("mlu")
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(a.stride() == b.stride())
            self.assertTrue(a.storage_offset() == b.storage_offset())
            self.assertTensorsEqual(a, b.cpu(), 0.0, use_MSE=True)

            input_cpu = torch.randn(shape).permute(order).contiguous()
            b.copy_(input_cpu)
            self.assertEqual(b.dtype, input_cpu.dtype)
            self.assertTensorsEqual(
                input_cpu.float(), b.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_non_overlapping_and_dense_D2H(self):
        shape_order_list = [
            [(3, 3), (1, 0)],
            [(5, 7), (1, 0)],
            [(3, 3, 3), (1, 2, 0)],
            [(3, 5, 7), (1, 0, 2)],
            [(32, 13, 4, 50), (2, 3, 0, 1)],
            [(3, 3, 3, 3), (1, 3, 2, 0)],
            [(3, 3, 3, 3, 3), (4, 1, 2, 0, 3)],
            [(2, 3, 5, 7, 11), (3, 1, 4, 0, 2)],
        ]

        for shape, order in shape_order_list:
            data = torch.randn(shape, dtype=torch.float)
            b = data.to("mlu").permute(order)
            a = b.cpu()
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(a.stride() == b.stride())
            self.assertTrue(a.storage_offset() == b.storage_offset())
            self.assertTensorsEqual(a, b.cpu(), 0.0, use_MSE=True)

            input_mlu = torch.randn(shape).permute(order).contiguous().to("mlu")
            a.copy_(input_mlu)
            self.assertEqual(a.dtype, input_mlu.dtype)
            self.assertTensorsEqual(
                a.float(), input_mlu.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_non_overlapping_and_dense_D2D(self):
        shape_order_list = [
            [(3, 3), (3, 3), (1, 0), (1, 0)],
            [(5, 7), (5, 7), (1, 0), (1, 0)],
            [(3, 3, 3), (3, 3, 3), (1, 2, 0), (2, 0, 1)],
            [(3, 5, 7), (7, 3, 5), (1, 0, 2), (2, 1, 0)],
            [(32, 13, 4, 50), (4, 32, 13, 50), (2, 3, 0, 1), (0, 3, 1, 2)],
            [(3, 3, 3, 3), (3, 3, 3, 3), (1, 3, 2, 0), (3, 0, 2, 1)],
            [(3, 3, 3, 3, 3), (3, 3, 3, 3, 3), (4, 1, 2, 0, 3), (2, 4, 3, 0, 1)],
            [(2, 3, 5, 7, 11), (2, 7, 3, 5, 11), (3, 1, 4, 0, 2), (1, 2, 4, 0, 3)],
        ]

        for shape1, shape2, order1, order2 in shape_order_list:
            data1 = torch.randn(shape1, dtype=torch.float)
            data2 = torch.randn(shape2, dtype=torch.float)
            data_mlu1 = data1.to("mlu").permute(order1)
            data_mlu2 = data2.to("mlu").permute(order2)
            self.assertEqual(data_mlu1.dtype, data_mlu2.dtype)
            self.assertEqual(data_mlu1.shape, data_mlu2.shape)

            data_mlu1.copy_(data_mlu2)
            self.assertTensorsEqual(data_mlu1.cpu(), data_mlu2.cpu(), 0.0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("64GB")
    def test_copy_H2D_large(self):
        in_shape = (5, 1024, 1024, 1024)
        dtype_list = [torch.half, torch.bool]
        for dtype in dtype_list:
            a = self.to_mlu(torch.randn(in_shape, dtype=torch.float))
            a_data_ptr = a.data_ptr()
            b = torch.randn(in_shape).to(dtype)
            a.copy_(b)
            self.assertEqual(b.dtype, dtype)
            self.assertEqual(a_data_ptr, a.data_ptr())
            self.assertEqual(a.dtype, torch.float)
            self.assertTensorsEqual(a.cpu(), b, 0.0, use_MSE=True)


if __name__ == "__main__":
    run_tests()
