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
    read_card_info,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestUpsampleNearest1dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_nearest1d(self):
        mode_list = ["nearest", "nearest-exact"]
        shape_list = [(3, 4, 5), (3, 10, 12), (1, 1, 19)]
        scale_factor_list = [0.4, 1, 2, 2.5]
        type_list = [torch.float32, torch.float16]
        func_list = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        param_list = [mode_list, shape_list, scale_factor_list, type_list, func_list]
        for mode, shape, scale_factor, dtype, func in itertools.product(*param_list):
            m = nn.Upsample(scale_factor=scale_factor, mode=mode)
            x = torch.randn(shape, dtype=dtype)
            out_cpu = m(func(x.float()))
            out_mlu = m(func(x.mlu()))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

            m = nn.Upsample(size=(34), scale_factor=None, mode=mode)
            out_cpu = m(func(x.float()))
            out_mlu = m(func(x.mlu()))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

            m = nn.Upsample(
                scale_factor=scale_factor, mode=mode, recompute_scale_factor=True
            )
            out_cpu = m(func(x.float()))
            out_mlu = m(func(x.mlu()))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_nearest1d_bp(self):
        mode_list = ["nearest", "nearest-exact"]
        shape_list = [(3, 4, 5), (3, 10, 12), (1, 1, 19)]
        scale_factor_list = [0.4, 1, 2, 2.5]
        type_list = [torch.float32, torch.float16]
        func_list = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        param_list = [mode_list, shape_list, scale_factor_list, type_list, func_list]
        for mode, shape, scale_factor, dtype, func in itertools.product(*param_list):
            m = nn.Upsample(scale_factor=scale_factor, mode=mode)
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

            m = nn.Upsample(size=(34), scale_factor=None, mode=mode)
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
                scale_factor=scale_factor, mode=mode, recompute_scale_factor=True
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
    def test_upsample_nearest1d_exception(self):
        shape = (2, 3, 4)
        m = nn.Upsample(scale_factor=2.5, mode="nearest")
        x_mlu = torch.randn(shape).to(torch.uint8).mlu()
        ref_msg = f"not implemented for 'Byte'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu = m(x_mlu)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_upsample_nearest1d_bfloat16(self):
        shape_list = [(3, 4, 5), (3, 10, 12)]
        type_list = [
            torch.bfloat16,
        ]
        func_list = [lambda x: x, self.convert_to_channel_last, self.to_non_dense]
        param_list = [shape_list, type_list, func_list]
        for shape, dtype, func in itertools.product(*param_list):
            m = nn.Upsample(scale_factor=2.5, mode="nearest")
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
    @largeTensorTest("73GB")
    def test_upsample_nearest1d_large(self):
        shape = (4 * 1025, 1024, 1024)
        dtype = torch.float
        m = nn.Upsample(scale_factor=2, mode="nearest")
        x = torch.randn(shape, dtype=dtype)
        out_cpu = m(x.float())
        out_mlu = m(x.mlu())
        # TODO: cnnlInterp_v3::nearest has accuracy problem when input large tensor
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("57GB")
    def test_upsample_nearest1d_bp_large(self):
        # TODO: see CNNLCORE-18738
        # [CNNL] [Error]:[cnnlInterpBackward_v3] overflow max supported tensor num 2147483647,
        # now tensor's total num is 4299161600.
        ref_msg = "CNNL error: CNNL_STATUS_NOT_SUPPORTED"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            shape = (1025, 1024, 1024)
            dtype = torch.float
            m = nn.Upsample(scale_factor=4, mode="nearest")
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
