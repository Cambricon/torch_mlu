from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
from torch import nn
from torch.nn import Parameter

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)

logging.basicConfig(level=logging.DEBUG)


class LayerNormModel(nn.Module):  # pylint: disable=W0223
    def __init__(self, nout, eps=1e-5):
        super(LayerNormModel, self).__init__()
        self.layernorm = torch.nn.LayerNorm(nout, eps=eps)

    def forward(self, x):
        y = self.layernorm(x)
        return y


class TestLayerNormOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm(self):
        gen_params = [
            ((256), (1, 199, 256)),
            ((1024), (37, 104, 1024)),
            ((1024), (32, 1, 1024)),
            ((2, 1024), (32, 2, 1024)),
            ((1024), (32, 1, 1, 1024)),
        ]

        for param in gen_params:
            input_ = torch.randn(param[1], dtype=torch.float)
            input_mlu = copy.deepcopy(input_).mlu()
            model = LayerNormModel(param[0])
            layer_norm = model
            layer_norm_mlu = copy.deepcopy(model).mlu()
            out_cpu = layer_norm(input_)
            out_mlu = layer_norm_mlu(input_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm_custom_weight_bias(self):
        gen_params = [
            ((256,), (1, 199, 256)),
            ((1024,), (37, 104, 1024)),
            ((1024,), (32, 1, 1024)),
            ((2, 1024), (32, 2, 1024)),
            ((1024,), (32, 1, 1, 1024)),
        ]

        for param in gen_params:
            input_ = torch.randn(param[1], dtype=torch.float)
            input_mlu = copy.deepcopy(input_).mlu()
            out_cpu = nn.functional.layer_norm(input_, param[0])
            out_mlu = nn.functional.layer_norm(input_mlu, param[0])
            weight_ = torch.randn(param[0], dtype=torch.float)
            weight_mlu = copy.deepcopy(weight_).mlu()
            custom_weight_out_cpu = nn.functional.layer_norm(
                input_, param[0], weight=weight_
            )
            custom_weight_out_mlu = nn.functional.layer_norm(
                input_mlu, param[0], weight=weight_mlu
            )
            bias_ = torch.randn(param[0], dtype=torch.float)
            bias_mlu = copy.deepcopy(bias_).mlu()
            custom_bias_out_cpu = nn.functional.layer_norm(input_, param[0], bias=bias_)
            custom_bias_out_mlu = nn.functional.layer_norm(
                input_mlu, param[0], bias=bias_mlu
            )
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                custom_weight_out_cpu, custom_weight_out_mlu.cpu(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                custom_bias_out_cpu, custom_bias_out_mlu.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm_not_dense(self):
        gen_params = [
            ((256), (1, 199, 256 * 2)),
            ((1024), (37, 104, 1024 * 2)),
            ((1024), (32, 1, 1024 * 2)),
            ((2, 1024), (32, 2, 1024 * 2)),
            ((1024), (32, 1, 1, 1024 * 2)),
        ]

        for param in gen_params:
            input_ = torch.randn(param[1], dtype=torch.float)
            input_cpu = input_[..., : int(param[1][-1] / 2)]
            input_mlu = copy.deepcopy(input_).mlu()[..., : int(param[1][-1] / 2)]
            model = LayerNormModel(param[0])
            layer_norm = model
            layer_norm_mlu = copy.deepcopy(model).mlu()
            out_cpu = layer_norm(input_cpu)
            out_mlu = layer_norm_mlu(input_mlu)

            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())
            self.assertTrue(input_cpu.size() == input_mlu.size())
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm_channel_last(self):
        gen_params = [
            ((256), (20, 1, 199, 256)),
            ((1024), (37, 5, 104, 1024)),
            ((2, 1024), (32, 32, 2, 1024)),
            ((1024), (32, 1, 1, 1024)),
        ]

        for param in gen_params:
            input_ = torch.randn(param[1], dtype=torch.float)
            input_cpu = input_.to(memory_format=torch.channels_last)
            input_mlu = (
                copy.deepcopy(input_).mlu().to(memory_format=torch.channels_last)
            )
            model = LayerNormModel(param[0])
            layer_norm = model
            layer_norm_mlu = copy.deepcopy(model).mlu()
            out_cpu = layer_norm(input_cpu)
            out_mlu = layer_norm_mlu(input_mlu)

            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())
            self.assertTrue(input_cpu.size() == input_mlu.size())
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_layer_norm_bfloat16(self):
        # Using cpu float test with mlu bfloat16
        # Root reason for acc diff is cpu using bfloat16 to accumulate mean
        # and std. GPU and MLU using float to accumulate mean and std.
        # Already submited a issue to pytorch community for this.
        gen_params = [((1024), (32, 1, 1, 1024))]
        for param in gen_params:
            input_ = torch.randn(param[1], dtype=torch.bfloat16)
            input_cpu = input_.float()
            input_mlu = copy.deepcopy(input_).mlu()
            model = LayerNormModel(param[0])
            model_cpu = copy.deepcopy(model)
            model_mlu = model.mlu()
            layer_norm = model_cpu.bfloat16().float()
            layer_norm_mlu = model_mlu.bfloat16()
            out_cpu = layer_norm(input_cpu)
            out_mlu = layer_norm_mlu(input_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("65GB")
    def test_layer_norm_large(self):
        gen_params = [((2), (1, 4096 * 48 * 13725, 2))]

        for param in gen_params:
            input_ = torch.randn(param[1], dtype=torch.float)
            input_mlu = copy.deepcopy(input_)
            layer_norm = LayerNormModel(param[0])
            layer_norm_mlu = LayerNormModel(param[0]).mlu()
            out_cpu = layer_norm(input_)
            out_mlu = layer_norm_mlu(input_mlu.mlu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
