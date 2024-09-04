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
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestUpsampleNearest2dOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_upsample_nearest2d(self):
        mode_list = ["nearest", "nearest-exact"]
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12), (1, 1, 19, 19)]
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

            m = nn.Upsample(size=(20, 34), scale_factor=None, mode=mode)
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
    def test_upsample_nearest2d_bp(self):
        # TODO(CNNLCORE-19092): uncomment after cnnl_v1.27.0
        mode_list = ["nearest"]  # , 'nearest-exact']
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12), (1, 1, 19, 19)]
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

            m = nn.Upsample(size=(20, 34), scale_factor=None, mode=mode)
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
    def test_upsample_nearest2d_exception(self):
        shape = (2, 3, 4, 5)
        m = nn.Upsample(scale_factor=2.5, mode="nearest")
        x_mlu = torch.randn(shape).to(torch.uint8).mlu()
        ref_msg = f"not implemented for 'Byte'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu = m(x_mlu)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_upsample_nearest2d_bfloat16(self):
        shape_list = [(2, 3, 4, 5), (3, 3, 10, 12)]
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


if __name__ == "__main__":
    unittest.main()
