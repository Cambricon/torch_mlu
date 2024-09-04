from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
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
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_prod_dtype_int32(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100)]

        x = torch.tensor(5, dtype=torch.int32)
        out_cpu = torch.prod(x)
        out_mlu = torch.prod(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=True)

        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randint(0, 3, shape)
                out_cpu = x.prod(item[1], keepdim=item[0])
                out_mlu = self.to_device(x).prod(item[1], keepdim=item[0])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )

                x = torch.randint(0, 3, shape)
                out_cpu = torch.prod(x)
                out_mlu = torch.prod(self.to_device(x))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_dim(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = x.prod(item[1], keepdim=item[0])
                out_mlu = self.to_device(x).prod(item[1], keepdim=item[0])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_prod(self):
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.prod(x)
            out_mlu = torch.prod(self.to_device(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_channel_last(self):
        x = torch.randn((2, 128, 10, 6), dtype=torch.float).to(
            memory_format=torch.channels_last
        )
        out_cpu = torch.prod(x)
        out_mlu = torch.prod(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

        # test not dense
        out_cpu = torch.prod(x[..., :2])
        out_mlu = torch.prod(self.to_device(x)[..., :2])
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.prod(x)
        out_mlu = torch.prod(self.to_device(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_out(self):
        type_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(type_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.randn(1)
                out_mlu = self.to_device(out_cpu)
                x_mlu = self.to_device(x)
                torch.prod(x, item[1], keepdim=item[0], out=out_cpu)
                torch.prod(x_mlu, item[1], keepdim=item[0], out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_empty(self):
        x = torch.randn(1, 0, 1)
        out_mlu = x.to("mlu").prod()
        out_cpu = x.prod()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_opt_dtype(self):
        shape = (2, 4, 2)
        type_list = [torch.int, torch.short, torch.int8, torch.long, torch.uint8]
        opt_list = [torch.int, torch.float, torch.double]
        for t, opttype in product(type_list, opt_list):
            x = (torch.randn(shape, dtype=torch.float) * 20).to(t)
            out_cpu = x.prod(dim=1, keepdim=True, dtype=opttype)
            out_mlu = x.to("mlu").prod(dim=1, keepdim=True, dtype=opttype)
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
        shape = (2, 2, 2)
        for t, opttype in product(type_list, opt_list):
            x = (torch.randn(shape, dtype=torch.float) * 5).to(t)
            out_cpu = x.prod(dtype=opttype)
            out_mlu = x.to("mlu").prod(dtype=opttype)

            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_backward(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x_mlu = self.to_device(x)

                out_cpu = torch.prod(x, item[1], keepdim=item[0])
                grad = torch.randn(out_cpu.shape)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                out_cpu.backward(grad)
                x_grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()

                out_mlu = torch.prod(x_mlu, item[1], keepdim=item[0])
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x.grad.float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_prod_exception(self):
        dtype = torch.float
        other_dtype = torch.float64
        x = torch.ones((3, 4, 5), device="mlu", dtype=dtype)
        out = torch.ones((3, 4, 5), device="mlu", dtype=dtype)
        msg = "Expected out tensor to have dtype double, but got float instead"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.prod(x, 0, out=out, dtype=other_dtype)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_prod_large(self):
        shape_list = [(48, 4096, 13725), (1, 4096 * 48 * 13725)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.prod(x)
            out_mlu = torch.prod(self.to_device(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_prod_bfloat16(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
                x_mlu = self.to_device(x)

                out_cpu = torch.prod(x.float(), item[1], keepdim=item[0])
                # torch.prod backward op is torch.cumprod,which cnnl kernel bfloat16 is not supported yet.
                # grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
                # grad_mlu = copy.deepcopy(grad).to('mlu')
                # out_cpu.backward(grad.float())
                # x_grad_cpu = copy.deepcopy(x.grad)
                # x.grad.zero_()

                out_mlu = torch.prod(x_mlu, item[1], keepdim=item[0])
                # out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("24GB")
    def test_prod_large_bfloat16(self):
        shape_list = [(48, 4096, 13725)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.bfloat16)
            out_cpu = torch.prod(x.float())
            out_mlu = torch.prod(self.to_device(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
