"""
test_reciprocal
"""
from __future__ import print_function

import sys
import os
import unittest
import logging
import copy

import torch
import torch_mlu

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestReciprocalOp(TestCase):
    """
    test-reciprocal
    """

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal(self):
        """
        test_reciprocal
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (1, 3, 2, 2), (3, 2, 4, 4), (3, 20), (4)]:
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                    if isinstance(shape1, int) or (
                        len(shape1) != 4 and memory_format == torch.channels_last
                    ):
                        continue
                    x_cpu = (
                        torch.rand(shape1, dtype=data_type).to(
                            memory_format=memory_format
                        )
                        + 0.00005
                    )
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type)

                    out_cpu = torch.reciprocal(x_cpu)
                    out_mlu = torch.reciprocal(x_mlu)

                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                    )

                    x_cpu = torch.rand(shape1, dtype=data_type) + 0.00005
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type)

                    out_cpu = 1 / x_cpu
                    out_mlu = 1 / x_mlu

                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_inplace(self):
        """
        test_reciprocal_inplace
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (1, 3, 2, 2), (3, 2, 4, 4), (3, 20), (4)]:
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                    if isinstance(shape1, int) or (
                        len(shape1) != 4 and memory_format == torch.channels_last
                    ):
                        continue
                    x_cpu = (
                        torch.rand(shape1, dtype=data_type).to(
                            memory_format=memory_format
                        )
                        + 0.00005
                    )
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                    x_mlu_ptr = x_mlu.data_ptr()

                    x_cpu.reciprocal_()
                    x_mlu.reciprocal_()

                    self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(
                        x_cpu.float(), x_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_out(self):
        """
        test_reciprocal_out
        """
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape1 in [(), (1, 3, 2, 2), (3, 2, 4, 4), (3, 20), (4)]:
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                    if isinstance(shape1, int) or (
                        len(shape1) != 4 and memory_format == torch.channels_last
                    ):
                        continue
                    x_cpu = (
                        torch.rand(shape1, dtype=data_type).to(
                            memory_format=memory_format
                        )
                        + 0.00005
                    )
                    x_mlu = self.to_mlu_dtype(x_cpu, data_type)  # pylint: disable=W0612
                    x_mlu_ptr = x_mlu.data_ptr()

                    out_cpu = torch.zeros(shape1, dtype=data_type)
                    out_mlu = torch.zeros(shape1, dtype=data_type).to("mlu")
                    torch.reciprocal(x_cpu, out=out_cpu)
                    torch.reciprocal(x_mlu, out=out_mlu)

                    self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float) + 0.00005
            x_mlu = x.mlu()
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            out_cpu = x.reciprocal()
            out_mlu = self.to_mlu(x).reciprocal()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_inplace_not_dense(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float) + 0.00005
            x_mlu = copy.deepcopy(x).mlu()
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            x_mlu = copy.deepcopy(x).mlu()
            x.reciprocal_()
            x_mlu.reciprocal_()
            self.assertTensorsEqual(x, x_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.rand(shape_list[i], dtype=torch.float) + 0.00005
            out = torch.rand(shape_list[i], dtype=torch.float) + 0.00005
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.reciprocal(x, out=out)
            torch.reciprocal(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_floating_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.rand((2, 3, 4, 5, 6), dtype=torch.half) + 0.00005
            x_mlu = self.to_mlu_dtype(x, dtype)
            x = x.float()
            x.reciprocal_()
            x_mlu.reciprocal_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_integral_dtype(self):
        dtype_list = [torch.uint8, torch.int8, torch.short, torch.int, torch.long]
        for dtype in dtype_list:
            x = torch.testing.make_tensor((2, 3, 4, 5, 6), dtype=dtype, device="cpu")
            x_mlu = x.to("mlu")
            output_cpu = torch.reciprocal(x)
            output_mlu = torch.reciprocal(x_mlu)
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu(), 3e-3, allow_inf=True, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_reciprocal_backward(self):
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
                out_cpu = torch.reciprocal(x_0)
                out_mlu = torch.reciprocal(x_mlu)

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
    def test_reciprocal_large(self):
        dtype_list = [(torch.half, 3e-3)]
        for data_type, err in dtype_list:
            for shape in [(5, 1024, 1024, 1024)]:
                x_cpu = torch.rand(shape, dtype=data_type) + 0.00005
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)

                out_cpu = torch.reciprocal(x_cpu)
                out_mlu = torch.reciprocal(x_mlu)

                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_reciprocal_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 3, 4, 6), dtype=torch.bfloat16, device="cpu"
        )
        left_cpu = torch.nn.Parameter(left)
        left_mlu = torch.nn.Parameter(left.mlu())
        out_cpu = torch.reciprocal(left_cpu)
        out_mlu = torch.reciprocal(left_mlu)
        grad = torch.randn_like(out_cpu)
        out_cpu.backward(grad)
        out_mlu.backward(grad.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, allow_inf=True, use_MSE=True
        )
        self.assertTensorsEqual(
            left_cpu.grad.float(),
            left_mlu.grad.cpu().float(),
            0.003,
            allow_inf=True,
            use_MSE=True,
        )


if __name__ == "__main__":
    run_tests()
