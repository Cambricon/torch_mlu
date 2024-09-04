from __future__ import print_function
import logging
import sys
import os
import unittest
import random
import numpy as np
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (  # pylint: disable=C0413,C0411
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    read_card_info,
    largeTensorTest,
)

TEST_BFLOAT16 = read_card_info()
torch.manual_seed(6503)


class TestCastOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_all_type_cast(self):
        shape = (2, 3, 4)
        type_list = [
            torch.half,
            torch.float,
            torch.double,
            torch.int,
            torch.short,
            torch.int8,
            torch.bool,
            torch.long,
            torch.uint8,
            torch.complex64,
            torch.chalf,
            torch.cdouble,
        ]
        for ori_t in type_list:
            x = torch.testing.make_tensor(shape, dtype=ori_t, device="cpu")
            for tar_t in type_list:
                out_cpu = x.to(tar_t)
                out_mlu = self.to_mlu(x).to(tar_t)
                if not out_cpu.is_complex():
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                else:
                    self.assertTensorsEqual(
                        out_cpu.imag, out_mlu.cpu().imag, 0.0, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        out_cpu.real, out_mlu.cpu().real, 0.0, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_large_value_cast(self):
        x = torch.testing.make_tensor((2,), dtype=torch.long, device="cpu")
        x_mlu = x.mlu()
        value = pow(2, 31) - 1
        x.fill_(value)
        x_mlu.fill_(value)
        type_list = [
            torch.int,
            torch.short,
            torch.int8,
            torch.bool,
            torch.uint8,
            torch.half,
            torch.float,
            torch.double,
            torch.long,
        ]
        for tar_t in type_list:
            cpu_out = x.to(tar_t)
            mlu_out = x_mlu.to(tar_t)
            self.assertEqual(cpu_out, mlu_out.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_uint8_cast(self):
        shape = (2, 3, 4)
        # TODO(shangang): half and float cast to uint8 failed by cnnl op bug.
        type_list = [torch.int, torch.short, torch.int8, torch.bool]
        for ori_t in type_list:
            x = torch.randn(shape, dtype=torch.float).to(ori_t)
            t = torch.uint8
            out_cpu = x.to(t)
            out_mlu = self.to_mlu(x).to(t)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_cast_permute(self):
        type_list = [torch.half, torch.float, torch.double]
        for ori_t in type_list:
            for in_shape in [
                (8, 224, 224),
                (1, 1, 1, 1),
                (1, 3, 16, 16, 4),
                (1, 3, 16, 16, 3, 6),
                (1, 3, 16, 16, 4, 15, 8),
            ]:
                input_ = torch.randn(in_shape, dtype=ori_t)
                size = np.arange(len(in_shape))
                random.shuffle(size)
                input_mlu = input_.to("mlu")
                input_ = torch.permute(input_, tuple(size))
                input_mlu = torch.permute(input_mlu, tuple(size))

                t = torch.uint8
                out_cpu = input_.to(t)
                out_mlu = input_mlu.to(t)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_cast_channel_last_not_dense_last(self):
        shapes_list = [(64, 3, 7, 7), (14, 7, 7, 7), (3, 4, 5), (3, 3, 4), (5, 5, 5, 5)]
        for shape1 in shapes_list:
            input = torch.randn(shape1, dtype=torch.float)
            if input.dim() == 4:
                input = input.to(memory_format=torch.channels_last)
            input_mlu = input.to("mlu")

            # channels_last
            output_cpu = input.to(torch.int)
            output_mlu = input_mlu.to(torch.int)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.00, use_MSE=True)
            # not dense
            output_cpu_not_dense = input[:, :2].to(torch.int)
            output_mlu_not_dense = input_mlu[:, :2].to(torch.int)
            self.assertTensorsEqual(
                output_cpu_not_dense, output_mlu_not_dense.cpu(), 0.00, use_MSE=True
            )

        # test inplace not contiguous
        input = torch.randn(3, 4, 5)
        output = torch.randn(3, 4, 5).int()
        output_cpu = self.to_non_dense(output)
        output_mlu = self.to_non_dense(output.mlu())
        output_cpu.copy_(input)
        output_mlu.copy_(input.mlu())
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.00)

    # @unittest.skip("not test")
    @testinfo()
    def test_cast_overlap(self):
        shapes_list = [(2, 256, 200, 304), (3, 4, 5), (3, 4, 2, 1, 5)]
        for shape1 in shapes_list:
            input = torch.tensor(1.0).expand(shape1)
            input_mlu = torch.tensor(1.0).mlu().expand(shape1)
            output = input.int()
            output_mlu = input_mlu.int()
            self.assertTensorsEqual(output, output_mlu.cpu(), 0.00)

    @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "only run case of large tensor by `--large`")
    @largeTensorTest("46GB")
    def test_float_cast_large(self):
        shape = (5, 1024, 1024, 1024)
        type_list = [(torch.float, torch.half)]
        for type_t in type_list:
            src_type, dst_type = type_t
            x = torch.randn(shape, dtype=src_type)
            out_cpu = x.to(dst_type)
            out_mlu = self.to_mlu(x).to(dst_type)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_bfloat16_cast(self):
        shape = (2, 3, 4)
        dtype_list = [
            (torch.float, torch.bfloat16, 0),
            (torch.half, torch.bfloat16, 0),
            (torch.int, torch.bfloat16, 0),
            (torch.short, torch.bfloat16, 0),
            (torch.int8, torch.bfloat16, 0),
            (torch.uint8, torch.bfloat16, 0),
            (torch.double, torch.bfloat16, 0),
            (torch.long, torch.bfloat16, 0),
            (torch.bool, torch.bfloat16, 0),
            (torch.bfloat16, torch.bfloat16, 0),
            (torch.chalf, torch.bfloat16, 0),
            (torch.cfloat, torch.bfloat16, 0),
            (torch.bfloat16, torch.chalf, 0),
            (torch.bfloat16, torch.cfloat, 0),
            (torch.bfloat16, torch.cdouble, 0),
            (torch.cdouble, torch.bfloat16, 0),
        ]
        for dtype_err in dtype_list:
            x = torch.testing.make_tensor(shape, dtype=dtype_err[0], device="cpu")
            out_cpu = x.to(dtype_err[1])
            out_mlu = x.mlu().to(dtype_err[1])
            if out_cpu.is_complex():
                self.assertTensorsEqual(
                    out_cpu.imag, out_mlu.cpu().imag, dtype_err[2], use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu.real, out_mlu.cpu().real, dtype_err[2], use_MSE=True
                )
            else:
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), dtype_err[2], use_MSE=True
                )
            # test inverse datatype cast
            x = torch.testing.make_tensor(shape, dtype=dtype_err[1], device="cpu")
            out_cpu = x.to(dtype_err[0])
            out_mlu = x.mlu().to(dtype_err[0])
            if out_cpu.is_complex():
                self.assertTensorsEqual(
                    out_cpu.imag, out_mlu.cpu().imag, dtype_err[2], use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu.real, out_mlu.cpu().real, dtype_err[2], use_MSE=True
                )
            else:
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu(), dtype_err[2], use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_LARGETENSOR, "only run case of large tensor by `--large`")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("46GB")
    def test_cast_large_bfloat16(self):
        shape = (5, 1024, 1024, 1024)
        type_list = [(torch.float, torch.bfloat16)]
        for type_t in type_list:
            src_type, dst_type = type_t
            x = torch.randn(shape, dtype=src_type)
            out_cpu = x.to(dst_type)
            out_mlu = self.to_mlu(x).to(dst_type)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
