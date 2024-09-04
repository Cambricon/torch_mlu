from __future__ import print_function
import sys
import os
import unittest
import copy

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
import logging  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestErfOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_erf_contiguous(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for shape in [
            (2, 3),
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            for data_type, err in dtype_list:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.erf(x)
                out_mlu = torch.erf(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

                # test inplace operation
                x_cpu = copy.deepcopy(x)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.erf_(x_cpu)
                torch.erf_(x_mlu)
                self.assertTensorsEqual(x_cpu, x_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_erf_channel_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for shape in [(1, 3, 16, 16)]:
            for data_type, err in dtype_list:
                x = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                out_cpu = torch.erf(x)
                out_mlu = torch.erf(self.to_mlu_dtype(x, data_type))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

                # test inplace operation
                x_cpu = copy.deepcopy(x)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.erf_(x_cpu)
                torch.erf_(x_mlu)
                self.assertTensorsEqual(x_cpu, x_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_erf_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3), (torch.double, 3e-3)]
        for shape in [(1, 3, 16, 16)]:
            for data_type, err in dtype_list:
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.erf(x[:, :, :, :8])
                out_mlu = torch.erf(self.to_mlu_dtype(x, data_type)[:, :, :, :8])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

                # test inplace operation
                x_cpu = copy.deepcopy(x)
                x_mlu = self.to_mlu_dtype(x, data_type)
                torch.erf_(x_cpu)
                torch.erf_(x_mlu)
                self.assertTensorsEqual(x_cpu, x_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_erf_backward(self):
        for shape in [
            (2, 3),
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16),
            (1, 3, 16, 16, 3),
        ]:
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            out_cpu = torch.erf(x)
            out_mlu = torch.erf(self.to_mlu(x))
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x_cpu = copy.deepcopy(x)
            x.grad.zero_()
            out_mlu.backward(self.to_mlu(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(x, x_cpu, 0.003, use_MSE=True)

            # test inplace operation
            x.grad.zero_()
            x_mlu = self.to_mlu(x)
            torch.erf_(x_mlu)
            x_mlu.backward(self.to_mlu(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_erf_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            out_mlu = copy.deepcopy(out).to("mlu")
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.erf(x, out=out)
            torch.erf(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_erf_type(self):
        shape_list = [(1, 3, 16, 16)]
        type_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.short,
            torch.bool,
        ]
        for shape in shape_list:
            for type in type_list:
                x_cpu = torch.randn(shape).to(type)
                x_mlu = self.to_mlu(x_cpu)
                if type == torch.half:
                    x_cpu = x_cpu.float()
                out_cpu = torch.erf(x_cpu)
                out_mlu = torch.erf(x_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_erf_bfloat16(self):
        shape = [2, 3, 4]
        x = torch.randn(shape, dtype=torch.bfloat16)
        out_cpu = torch.erf(x)
        out_mlu = torch.erf(self.to_mlu(x))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
