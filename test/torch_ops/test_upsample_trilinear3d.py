from __future__ import print_function

import sys
import copy
import logging
import os
import itertools
import unittest
import torch
import torch.nn as nn
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestUpsampleTrilinear3dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_trilinear3d(self):
        shape_list = [(1, 2, 3, 4, 5), (6, 6, 6, 6, 6)]
        align_corners = [True, False]
        type_list = [torch.float32, torch.float16]
        func_list = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        param_list = [shape_list, align_corners, type_list, func_list]
        for shape, corner, dtype, func in itertools.product(*param_list):
            m = nn.Upsample(scale_factor=2.5, mode="trilinear", align_corners=corner)
            x = torch.randn(shape, requires_grad=True, dtype=dtype)
            x_mlu = copy.deepcopy(x)
            out_cpu = m(func(x.float()))
            out_mlu = m(func(x_mlu.mlu()))
            grad = torch.randn(out_cpu.shape, dtype=dtype)
            out_cpu.backward(grad.float())
            grad_cpu = x.grad
            out_mlu.backward(grad.mlu())
            grad_mlu = x_mlu.grad
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
            )

            m = nn.Upsample(
                size=(17, 20, 34),
                scale_factor=None,
                mode="trilinear",
                align_corners=corner,
            )
            out_cpu = m(func(x.float()))
            out_mlu = m(func(x_mlu.mlu()))
            grad = torch.randn(out_cpu.shape, dtype=dtype)
            out_cpu.backward(grad.float())
            grad_cpu = x.grad
            out_mlu.backward(grad.mlu())
            grad_mlu = x_mlu.grad
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
            )

            m = nn.Upsample(
                scale_factor=2.5,
                mode="trilinear",
                align_corners=corner,
                recompute_scale_factor=True,
            )
            out_cpu = m(func(x.float()))
            out_mlu = m(func(x_mlu.mlu()))
            grad = torch.randn(out_cpu.shape, dtype=dtype)
            out_cpu.backward(grad.float())
            grad_cpu = x.grad
            out_mlu.backward(grad.mlu())
            grad_mlu = x_mlu.grad
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_trilinear3d_exception(self):
        shape = (2, 3, 4, 5, 4)
        m = nn.Upsample(scale_factor=2.5, mode="trilinear", align_corners=True)
        x_mlu = torch.randn(shape).to(torch.uint8).mlu()
        ref_msg = f"not implemented for 'Byte'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu = m(x_mlu)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_upsample_trilinear3d_bfloat16(self):
        shape_list = [(1, 2, 3, 4, 5), (6, 6, 6, 6, 6)]
        align_corners = [True, False]
        type_list = [
            torch.bfloat16,
        ]
        func_list = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        param_list = [shape_list, align_corners, type_list, func_list]
        for shape, corner, dtype, func in itertools.product(*param_list):
            m = nn.Upsample(scale_factor=2.5, mode="trilinear", align_corners=corner)
            x = torch.randn(shape, requires_grad=True, dtype=dtype)
            x_mlu = copy.deepcopy(x)
            out_cpu = m(func(x.float()))
            out_mlu = m(func(x_mlu.mlu()))
            grad = torch.randn(out_cpu.shape, dtype=dtype)
            out_cpu.backward(grad.float())
            grad_cpu = x.grad
            out_mlu.backward(grad.mlu())
            grad_mlu = x_mlu.grad
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("45GB")
    def test_upsample_trilinear3d_large(self):
        # [CNNL] [Error]:[cnnlInterp_v3] overflow max supported tensor num 2147483647,
        # now tensor's total num is 8598323200.
        ref_msg = "CNNL error: CNNL_STATUS_NOT_SUPPORTED"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            shape = (2, 1025, 2, 512, 512)
            dtype = torch.float
            m = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            x = torch.randn(shape, requires_grad=True, dtype=dtype)
            x_mlu = copy.deepcopy(x)
            out_cpu = m(x.float())
            out_mlu = m(x_mlu.mlu())
            grad = torch.randn(out_cpu.shape, dtype=dtype)
            out_cpu.backward(grad.float())
            grad_cpu = x.grad
            out_mlu.backward(grad.mlu())
            grad_mlu = x_mlu.grad
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
