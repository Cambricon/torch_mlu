from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
from itertools import product
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
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestLogOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_log(self):
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (24, 4560, 3, 6, 20), (1), (0), ()]
        channel_first = [True, False]
        for shape in shape_list:
            for channel in channel_first:
                x = torch.rand(shape) + 0.0001
                out_cpu = torch.log(x)
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_mlu = torch.log(self.to_device(x))
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().contiguous(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_log_inplace(self):
        torch.manual_seed(0)
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (1)]
        for shape in shape_list:
            x = torch.rand(shape) + 0.0001
            x_cpu = self.convert_to_channel_last(x)
            x_mlu = self.convert_to_channel_last(self.to_device(x))
            x_cpu.log_()
            data_ptr_pre = x_mlu.data_ptr()
            x_mlu.log_()
            self.assertTensorsEqual(
                x_cpu, x_mlu.cpu().contiguous(), 0.003, use_MSE=True
            )
            self.assertEqual(data_ptr_pre, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_log_out(self):
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (1)]
        out_shape_list = [(24), (240, 8760), (1)]
        channel_first = [True, False]
        for shape, out_shape in zip(shape_list, out_shape_list):
            for channel in channel_first:
                x = torch.rand(shape) + 0.0001
                if channel is False:
                    x = self.convert_to_channel_last(x)
                out_cpu = torch.randn(out_shape)
                out_mlu = self.to_device(torch.randn(out_shape))
                torch.log(x, out=out_cpu)
                torch.log(self.to_device(x), out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log2(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape in shape_list:
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    out_cpu = torch.log2(x.float())
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_mlu = torch.log2(self.to_mlu_dtype(x, data_type))
                    self.assertTensorsEqual(
                        out_cpu.float(),
                        out_mlu.cpu().float().contiguous(),
                        err,
                        use_MSE=True,
                    )

                    x = torch.tensor([1.0017]).to(data_type)
                    out_cpu = torch.log2(x.float())
                    out_mlu = torch.log2(self.to_mlu_dtype(x, data_type))
                    self.assertTensorsEqual(
                        out_cpu.float(),
                        out_mlu.cpu().float().contiguous(),
                        err,
                        use_MSE=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_log2_inplace(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for data_type, err in dtype_list:
            for shape in shape_list:
                x = torch.rand(shape).to(data_type) + 1
                x_cpu = self.convert_to_channel_last(x)
                x_mlu = self.convert_to_channel_last(self.to_mlu_dtype(x, data_type))
                x_cpu = x_cpu.float()
                x_cpu.log2_()
                data_ptr_pre = x_mlu.data_ptr()
                x_mlu.log2_()
                self.assertTensorsEqual(
                    x_cpu.float(), x_mlu.cpu().float(), err, use_MSE=True
                )
                self.assertEqual(data_ptr_pre, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_log2_out(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        out_shape_list = [(12), (24), (240, 8760), (3, 2, 4), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape, out_shape in zip(shape_list, out_shape_list):
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_cpu = torch.randn(out_shape)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                    torch.log2(x.float(), out=out_cpu)
                    torch.log2(self.to_mlu_dtype(x, data_type), out=out_mlu)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_log10(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape in shape_list:
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    out_cpu = torch.log10(x.float())
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_mlu = torch.log10(self.to_mlu_dtype(x, data_type))
                    self.assertTensorsEqual(
                        out_cpu.float(),
                        out_mlu.cpu().float().contiguous(),
                        err,
                        use_MSE=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_log10_inplace(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for data_type, err in dtype_list:
            for shape in shape_list:
                x = torch.rand(shape).to(data_type) + 1
                x_cpu = self.convert_to_channel_last(x)
                x_mlu = self.convert_to_channel_last(self.to_mlu_dtype(x, data_type))
                x_cpu = x_cpu.float()
                x_cpu.log10_()
                data_ptr_pre = x_mlu.data_ptr()
                x_mlu.log10_()
                self.assertTensorsEqual(
                    x_cpu.float(), x_mlu.cpu().float(), err, use_MSE=True
                )
                self.assertEqual(data_ptr_pre, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_log10_out(self):
        shape_list = [(), (2, 3, 4), (24, 8760, 2, 5), (3, 2, 4, 5, 6, 1, 2), (1)]
        out_shape_list = [(12), (24), (240, 8760), (3, 2, 4), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        channel_first = [True, False]
        for data_type, err in dtype_list:
            for shape, out_shape in zip(shape_list, out_shape_list):
                for channel in channel_first:
                    x = torch.rand(shape).to(data_type) + 1
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_cpu = torch.randn(out_shape)
                    out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                    torch.log10(x.float(), out=out_cpu)
                    torch.log10(self.to_mlu_dtype(x, data_type), out=out_mlu)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_log1p(self):
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (24, 4560, 3, 6, 20), (1), (0), ()]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, shape, func in product(dtype_list, shape_list, func_list):
            data_type, err = dtype_err
            x = torch.rand(shape).to(data_type) + 0.0001
            out_cpu = torch.log1p(func(x.float()))
            out_mlu = torch.log1p(func(self.to_mlu_dtype(x, data_type)))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float().contiguous(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_log1p_inplace(self):
        torch.manual_seed(0)
        shape_list = [(2, 3, 4), (24, 8769, 2, 5), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, shape, func in product(dtype_list, shape_list, func_list):
            data_type, err = dtype_err
            x = torch.rand(shape).to(data_type) + 0.0001
            x_cpu = func(x)
            x_mlu = func(self.to_mlu_dtype(x, data_type))
            x_cpu = x_cpu.float()
            x_cpu.log1p_()
            data_ptr_pre = x_mlu.data_ptr()
            x_mlu.log1p_()
            self.assertTensorsEqual(
                x_cpu, x_mlu.cpu().float().contiguous(), err, use_MSE=True
            )
            self.assertEqual(data_ptr_pre, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_log1p_out(self):
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (1)]
        out_shape_list = [(24), (240, 8760), (1)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, func in product(dtype_list, func_list):
            for shape, out_shape in zip(shape_list, out_shape_list):
                data_type, err = dtype_err
                x = torch.rand(shape).to(data_type) + 0.0001
                out_cpu = torch.randn(out_shape)
                out_mlu = self.to_mlu_dtype(torch.randn(out_shape), data_type)
                torch.log1p(func(x.float()), out=out_cpu)
                torch.log1p(func(self.to_mlu_dtype(x, data_type)), out=out_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log_all_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        func_list = [torch.log1p, torch.log, torch.log10, torch.log2]
        for i in range(4):
            for func in func_list:
                x = torch.rand(shape_list[i], dtype=torch.float) + 1
                out = torch.rand(shape_list[i], dtype=torch.float)
                x_mlu = copy.deepcopy(x).to("mlu")
                out_mlu = copy.deepcopy(out).to("mlu")
                x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
                x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                    permute_shape[i]
                )
                func(x, out=out)
                func(x_mlu, out=out_mlu)
                self.assertTrue(out.stride() == out_mlu.stride())
                self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
                self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log2_exception(self):
        x_mlu = torch.rand(1, 16).int().to("mlu")
        ref_msg = "result type Float can't be cast to the desired output type Int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.log2_()
        out_mlu = torch.zeros(1).int().to("mlu")
        ref_msg = "result type Float can't be cast to the desired output type Int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.log2(x_mlu, out=out_mlu)
        x_mlu = torch.rand(1, 16).to("mlu")
        ref_msg = "result type Float can't be cast to the desired output type Int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.log2(x_mlu, out=out_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_log_floating_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.rand((2, 3, 4, 5, 6), dtype=torch.half) + 0.00005
            x_mlu = self.to_mlu_dtype(x, dtype)
            x = x.float()
            x.log_()
            x_mlu.log_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log_integral_dtype(self):
        dtype_list = [torch.uint8, torch.int8, torch.short, torch.int, torch.long]
        for dtype in dtype_list:
            x = torch.testing.make_tensor((2, 3, 4, 5, 6), dtype=dtype, device="cpu")
            x_mlu = x.to("mlu")
            output_cpu = torch.log(x)
            output_mlu = torch.log(x_mlu)
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu(), 3e-3, allow_inf=True, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_log_backward(self):
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
                out_cpu = torch.log(x_0)
                out_mlu = torch.log(x_mlu)
                out_cpu.backward(torch.ones_like(out_cpu))
                out_mlu.backward(torch.ones_like(out_mlu))
                self.assertTensorsEqual(
                    x_0.grad, x_mlu.grad.cpu(), 0.003, allow_inf=True, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_log1p_backward(self):
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
                out_cpu = torch.log(x_0)
                out_mlu = torch.log(x_mlu)
                out_cpu.backward(torch.ones_like(out_cpu))
                out_mlu.backward(torch.ones_like(out_mlu))
                self.assertTensorsEqual(
                    x_0.grad, x_mlu.grad.cpu(), 0.003, allow_inf=True, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_log1p_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        dtype_list = [(torch.half, 3e-2)]
        func_list = [self.convert_to_channel_last, lambda x: x]
        for dtype_err, shape, func in product(dtype_list, shape_list, func_list):
            data_type, err = dtype_err
            x = torch.rand(shape).to(data_type) + 0.0001
            out_cpu = torch.log1p(func(x.float()))
            out_mlu = torch.log1p(func(self.to_mlu_dtype(x, data_type)))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float().contiguous(), err, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("46GB")
    def test_log_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        for shape in shape_list:
            x = torch.rand(shape) + 0.0001
            out_cpu = torch.log(x)
            out_mlu = torch.log(self.to_device(x))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().contiguous(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_log1p_bfloat16(self):
        shape_list = [(2, 3, 4), (24, 8760, 2, 5), (24, 4560, 3, 6, 20), (1), (0), ()]
        dtype_list = [(torch.bfloat16, 3e-3)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, shape, func in product(dtype_list, shape_list, func_list):
            data_type, err = dtype_err
            x = torch.rand(shape).to(data_type) + 0.0001
            out_cpu = torch.log1p(func(x))
            out_mlu = torch.log1p(func(self.to_mlu(x)))
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().contiguous(), err, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
