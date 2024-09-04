# pylint: disable=W0223,R0201,C0413,C0411,C0301
from itertools import product
import sys
import os
import unittest
import logging
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_BFLOAT16,
    TEST_LARGETENSOR,
    largeTensorTest,
)


class TestLerpOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_tensor(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, shape, func in product(dtype_list, shape_list, func_list):
            data_type, err = dtype_err
            a = torch.randn(shape, dtype=data_type)
            b = torch.randn(shape, dtype=data_type)
            w = torch.randn(shape, dtype=data_type)
            a_mlu, b_mlu, w_mlu = (
                func(a.to("mlu")),
                func(b.to("mlu")),
                func(w.to("mlu")),
            )
            a_cpu, b_cpu, w_cpu = func(a), func(b), func(w)
            out_cpu = torch.lerp(a_cpu.float(), b_cpu.float(), w_cpu.float())
            out_mlu = torch.lerp(a_mlu, b_mlu, w_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_scalar(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, shape, func in product(dtype_list, shape_list, func_list):
            data_type, err = dtype_err
            a = torch.randn(shape, dtype=data_type)
            b = torch.randn(shape, dtype=data_type)
            w = 10
            a_mlu, b_mlu = func(a.to("mlu")), func(b.to("mlu"))
            a_cpu, b_cpu = func(a), func(b)
            out_cpu = torch.lerp(a_cpu.float(), b_cpu.float(), w)
            out_mlu = torch.lerp(a_mlu, b_mlu, w)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_inplace_tensor(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for dtype_err, shape in product(dtype_list, shape_list):
            data_type, err = dtype_err
            a = torch.randn(shape, dtype=data_type)
            b = torch.randn(shape, dtype=data_type)
            w = torch.randn(shape, dtype=data_type)
            a_mlu, b_mlu, w_mlu = a.to("mlu"), b.to("mlu"), w.to("mlu")
            a = a.float()
            ptr1 = a_mlu.data_ptr()
            a.lerp_(b.float(), w.float())
            a_mlu.lerp_(b_mlu, w_mlu)
            ptr2 = a_mlu.data_ptr()
            self.assertEqual(ptr1, ptr2)
            self.assertTensorsEqual(a, a_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_inplace_scalar(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for dtype_err, shape in product(dtype_list, shape_list):
            data_type, err = dtype_err
            a = torch.randn(shape, dtype=data_type)
            b = torch.randn(shape, dtype=data_type)
            w = 10
            a_mlu, b_mlu = a.to("mlu"), b.to("mlu")
            a = a.float()
            ptr1 = a_mlu.data_ptr()
            a.lerp_(b.float(), w)
            a_mlu.lerp_(b_mlu, w)
            ptr2 = a_mlu.data_ptr()
            self.assertEqual(ptr1, ptr2)
            self.assertTensorsEqual(a, a_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_out_tensor(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for dtype_err, shape in product(dtype_list, shape_list):
            data_type, err = dtype_err
            a = torch.randn(shape, dtype=data_type)
            b = torch.randn(shape, dtype=data_type)
            w = torch.randn(shape, dtype=data_type)
            output = torch.randn(shape, dtype=data_type)
            a_mlu, b_mlu, w_mlu, output_mlu = (
                a.to("mlu"),
                b.to("mlu"),
                w.to("mlu"),
                output.to("mlu"),
            )
            output = output.float()
            torch.lerp(a.float(), b.float(), w.float(), out=output)
            torch.lerp(a_mlu, b_mlu, w_mlu, out=output_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_out_scalar(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-2)]
        for dtype_err, shape in product(dtype_list, shape_list):
            data_type, err = dtype_err
            a = torch.randn(shape, dtype=data_type)
            b = torch.randn(shape, dtype=data_type)
            w = 10
            output = torch.randn(shape, dtype=data_type)
            a_mlu, b_mlu, output_mlu = a.to("mlu"), b.to("mlu"), output.to("mlu")
            output = output.float()
            torch.lerp(a.float(), b.float(), w, out=output)
            torch.lerp(a_mlu, b_mlu, w, out=output_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_scalar_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            permute = permute_shape[i]
            a = torch.randn(shape_list[i])
            b = torch.randn(shape_list[i])
            w = 10
            output = torch.randn(shape_list[i])
            a_mlu, b_mlu, output_mlu = a.to("mlu"), b.to("mlu"), output.to("mlu")
            a, b, output = (
                a.permute(permute),
                b.permute(permute),
                output.permute(permute),
            )
            a_mlu, b_mlu, output_mlu = (
                a_mlu.permute(permute),
                b_mlu.permute(permute),
                output_mlu.permute(permute),
            )
            torch.lerp(a, b, w, out=output)
            torch.lerp(a_mlu, b_mlu, w, out=output_mlu)
            self.assertTrue(output.stride() == output_mlu.stride())
            self.assertTrue(output.storage_offset() == output_mlu.storage_offset())
            self.assertTensorsEqual(output, output_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_lerp_scalar_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            a = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            b = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            w = 10
            a_mlu, b_mlu = self.to_mlu_dtype(a, dtype), self.to_mlu_dtype(b, dtype)
            a, b = a.float(), b.float()
            output = torch.lerp(a, b, w)
            output_mlu = torch.lerp(a_mlu, b_mlu, w)
            self.assertTensorsEqual(
                output, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_lerp_scalar_bfloat16(self):
        dtype_list = [
            torch.bfloat16,
        ]
        for dtype in dtype_list:
            a = torch.randn((2, 3, 4, 5, 6), dtype=dtype)
            b = torch.randn((2, 3, 4, 5, 6), dtype=dtype)
            w = 10
            a_mlu, b_mlu = a.to("mlu"), b.to("mlu")
            output = torch.lerp(a, b, w)
            output_mlu = torch.lerp(a_mlu, b_mlu, w)
            self.assertTensorsEqual(output, output_mlu.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("53GB")
    def test_lerp_scalar_large(self):
        shape_list = [(4, 1025, 1024, 1024)]
        for shape in shape_list:
            a = torch.randn(shape)
            b = torch.randn(shape)
            w = 10
            a_mlu, b_mlu = a.to("mlu"), b.to("mlu")
            out_cpu = torch.lerp(a, b, w)
            out_mlu = torch.lerp(a_mlu, b_mlu, w)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    run_tests()
