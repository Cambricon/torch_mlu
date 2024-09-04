# pylint: disable=W0511
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
    def __init__(self, nout, weight, bias, eps=1e-5):
        super(LayerNormModel, self).__init__()
        self.layernorm = torch.nn.LayerNorm(nout, eps=eps)
        self.layernorm.weight = Parameter(weight)
        self.layernorm.bias = Parameter(bias)

    def forward(self, x):
        y = self.layernorm(x)
        return y


class TestLayerNormOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm_backward(self):
        # init: param | input: tensor
        gen_params = [
            ((1024), (37, 104, 1024)),
            ((500), (32, 100, 500)),
            ((4), (1, 4, 4)),
            ((1024), (32, 1, 1024)),
            ((2, 1024), (32, 2, 1024)),
            ((1024), (32, 1, 1, 1024)),
        ]
        for param in gen_params:
            weight = torch.randn(param[0], dtype=torch.float, requires_grad=True)
            bias = torch.randn(param[0], dtype=torch.float, requires_grad=True)
            weight_mlu = copy.deepcopy(weight)
            bias_mlu = copy.deepcopy(bias)
            input_ = torch.randn(param[1], dtype=torch.float, requires_grad=True)
            input_mlu = copy.deepcopy(input_)
            grad_cpu = torch.randn(input_.shape, dtype=torch.float)
            grad_mlu = copy.deepcopy(grad_cpu).mlu()
            layer_norm = LayerNormModel(param[0], weight, bias)
            layer_norm_mlu = LayerNormModel(param[0], weight_mlu, bias_mlu).mlu()
            out_cpu = layer_norm(input_)
            out_cpu.backward(grad_cpu)
            input_grad_cpu = copy.deepcopy(input_.grad)
            input_.grad.zero_()
            out_mlu = layer_norm_mlu(input_mlu.mlu())
            out_mlu.backward(grad_mlu)
            input_grad_mlu = copy.deepcopy(input_mlu.grad)
            input_mlu.grad.zero_()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_grad_cpu, input_grad_mlu.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm_backward_not_dense(self):
        # init: param | input: tensor
        gen_params = [
            ((1024 * 2), (37, 104, 1024 * 2)),
            ((500 * 2), (32, 100, 500 * 2)),
            ((4 * 2), (1, 4, 4 * 2)),
            ((1024 * 2), (32, 1, 1024 * 2)),
            ((1024 * 2), (32, 1, 1, 1024 * 2)),
        ]
        for param in gen_params:
            weight = torch.randn(param[0], dtype=torch.float)
            bias = torch.randn(param[0], dtype=torch.float)
            input_ = torch.randn(param[1], dtype=torch.float)
            grad = torch.randn(input_.shape, dtype=torch.float)
            weight_cpu = copy.deepcopy(weight)[
                ..., : int(param[0] / 2)
            ].requires_grad_()
            bias_cpu = copy.deepcopy(bias)[..., : int(param[0] / 2)].requires_grad_()
            input_cpu = copy.deepcopy(input_)[
                ..., : int(param[1][-1] / 2)
            ].requires_grad_()
            grad_cpu = copy.deepcopy(grad)[..., : int(param[1][-1] / 2)]
            weight_mlu = (
                copy.deepcopy(weight).mlu()[..., : int(param[0] / 2)].requires_grad_()
            )
            bias_mlu = (
                copy.deepcopy(bias).mlu()[..., : int(param[0] / 2)].requires_grad_()
            )
            input_mlu = (
                copy.deepcopy(input_)
                .mlu()[..., : int(param[1][-1] / 2)]
                .requires_grad_()
            )
            grad_mlu = copy.deepcopy(grad).mlu()[..., : int(param[1][-1] / 2)]

            layer_norm = LayerNormModel([int(param[0] / 2)], weight_cpu, bias_cpu)
            layer_norm_mlu = LayerNormModel(
                [int(param[0] / 2)], weight_mlu, bias_mlu
            ).mlu()
            out_cpu = layer_norm(input_cpu)
            out_cpu.backward(grad_cpu)
            input_grad_cpu = copy.deepcopy(input_cpu.grad)
            input_cpu.grad.zero_()
            out_mlu = layer_norm_mlu(input_mlu)
            out_mlu.backward(grad_mlu)
            input_grad_mlu = copy.deepcopy(input_mlu.grad)
            input_mlu.grad.zero_()
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())
            self.assertTrue(input_grad_cpu.size() == input_grad_mlu.size())
            self.assertTrue(input_grad_cpu.stride() == input_grad_mlu.stride())
            self.assertTrue(
                input_grad_cpu.storage_offset() == input_grad_mlu.storage_offset()
            )
            self.assertTrue(input_cpu.size() == input_mlu.size())
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_grad_cpu, input_grad_mlu.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm_backward_channel_last(self):
        # init: param | input: tensor
        gen_params = [
            ((256), (20, 1, 199, 256)),
            ((2, 1024), (32, 32, 2, 1024)),
            ((1024), (32, 1, 1, 1024)),
            ((1024), (37, 5, 104, 1024)),
        ]
        for param in gen_params:
            weight = torch.randn(param[0], dtype=torch.float)
            bias = torch.randn(param[0], dtype=torch.float)
            input_ = torch.randn(param[1], dtype=torch.float)
            grad = torch.randn(input_.shape, dtype=torch.float)
            weight_cpu = copy.deepcopy(weight).requires_grad_()
            bias_cpu = copy.deepcopy(bias).requires_grad_()
            input_cpu = (
                copy.deepcopy(input_)
                .to(memory_format=torch.channels_last)
                .requires_grad_()
            )
            grad_cpu = copy.deepcopy(grad).to(memory_format=torch.channels_last)

            weight_mlu = copy.deepcopy(weight).mlu().requires_grad_()
            bias_mlu = copy.deepcopy(bias).mlu().requires_grad_()
            input_mlu = (
                copy.deepcopy(input_)
                .mlu()
                .to(memory_format=torch.channels_last)
                .requires_grad_()
            )
            grad_mlu = copy.deepcopy(grad).mlu().to(memory_format=torch.channels_last)

            layer_norm = LayerNormModel(param[0], weight_cpu, bias_cpu)
            layer_norm_mlu = LayerNormModel(param[0], weight_mlu, bias_mlu).mlu()
            out_cpu = layer_norm(input_cpu)
            out_cpu.backward(grad_cpu)
            input_grad_cpu = copy.deepcopy(input_cpu.grad)
            input_cpu.grad.zero_()
            out_mlu = layer_norm_mlu(input_mlu)
            out_mlu.backward(grad_mlu)
            input_grad_mlu = copy.deepcopy(input_mlu.grad)
            input_mlu.grad.zero_()
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_cpu.storage_offset() == out_mlu.storage_offset())
            self.assertTrue(input_grad_cpu.size() == input_grad_mlu.size())
            self.assertTrue(input_grad_cpu.stride() == input_grad_mlu.stride())
            self.assertTrue(
                input_grad_cpu.storage_offset() == input_grad_mlu.storage_offset()
            )
            self.assertTrue(input_cpu.size() == input_mlu.size())
            self.assertTrue(input_cpu.stride() == input_mlu.stride())
            self.assertTrue(input_cpu.storage_offset() == input_mlu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_grad_cpu, input_grad_mlu.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_layer_norm_backward_module(self):
        # init: param | input: tensor | requires_grad: bool | elementwise_affine: bool
        gen_params = [
            ((1024), (37, 104, 1024), True, False),
            ((500), (32, 100, 500), False, True),
            ((4), (1, 4, 4), True, True),
            ((1024), (32, 1, 1024), False, False),
            ((2, 1024), (32, 2, 1024), False, True),
            ((2, 1024), (32, 2, 1024), True, False),
            ((1024), (32, 1, 1, 1024), True, False),
            ((1024), (0, 1, 1, 1024), True, False),
            ((500), (0, 100, 500), False, True),
            ((4), (0, 4, 4), True, True),
            ((1024), (0, 1, 1024), False, False),
        ]
        dtypes = [torch.double, torch.float]
        for param in gen_params:
            for t in dtypes:
                input_cpu = torch.randn(param[1], dtype=t, requires_grad=param[2])
                # backward need a leaf node in graph
                leaf_node_cpu = torch.randn(param[1], dtype=t, requires_grad=True)
                grad_cpu = torch.randn(input_cpu.shape, dtype=t)
                layer_norm = torch.nn.LayerNorm(
                    param[0], elementwise_affine=param[3]
                ).to(t)
                out_cpu = layer_norm(input_cpu)
                out_cpu = out_cpu + leaf_node_cpu
                out_cpu.backward(grad_cpu)

                leaf_node_grad_cpu = copy.deepcopy(leaf_node_cpu.grad)
                leaf_node_cpu.grad.zero_()
                if input_cpu.grad is not None:
                    input_grad_cpu = copy.deepcopy(input_cpu.grad)
                    input_cpu.grad.zero_()
                if layer_norm.weight is not None:
                    weight_grad_cpu = copy.deepcopy(layer_norm.weight.grad)
                    layer_norm.weight.grad.zero_()
                if layer_norm.bias is not None:
                    bias_grad_cpu = copy.deepcopy(layer_norm.bias.grad)
                    layer_norm.bias.grad.zero_()

                input_mlu = input_cpu.mlu()
                grad_mlu = copy.deepcopy(grad_cpu).mlu()
                leaf_node_mlu = leaf_node_cpu.mlu()
                layer_norm_mlu = layer_norm.mlu()
                out_mlu = layer_norm_mlu(input_mlu)
                out_mlu = out_mlu + leaf_node_mlu
                out_mlu.backward(grad_mlu)

                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                leaf_node_grad_mlu = copy.deepcopy(leaf_node_cpu.grad)
                self.assertTensorsEqual(
                    leaf_node_grad_cpu, leaf_node_grad_mlu, 0.003, use_MSE=True
                )
                if input_cpu.grad is not None:
                    input_grad_mlu = copy.deepcopy(input_cpu.grad)
                    self.assertTensorsEqual(
                        input_grad_cpu, input_grad_mlu, 0.003, use_MSE=True
                    )
                if layer_norm_mlu.weight is not None:
                    weight_grad_mlu = copy.deepcopy(layer_norm_mlu.weight.grad)
                    self.assertTensorsEqual(
                        weight_grad_cpu, weight_grad_mlu.cpu(), 0.003, use_MSE=True
                    )
                if layer_norm_mlu.bias is not None:
                    bias_grad_mlu = copy.deepcopy(layer_norm_mlu.bias.grad)
                    self.assertTensorsEqual(
                        bias_grad_cpu, bias_grad_mlu.cpu(), 0.003, use_MSE=True
                    )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_layer_norm_backward_bfloat16(self):
        # Using cpu float test with mlu bfloat16
        # Root reason for acc diff is cpu using bfloat16 to accumulate mean
        # and std. GPU and MLU using float to accumulate mean and std.
        # Already submited a issue to pytorch community for this.
        gen_params = [((1024), (37, 104, 1024))]
        for param in gen_params:
            weight = torch.randn(param[0], dtype=torch.float)
            bias = torch.randn(param[0], dtype=torch.float)
            weight_mlu = copy.deepcopy(weight).mlu().bfloat16()
            bias_mlu = copy.deepcopy(bias).mlu().bfloat16()
            input_ = torch.randn(param[1], dtype=torch.bfloat16)
            input = copy.deepcopy(input_).float()
            input_copy = copy.deepcopy(input_).mlu()
            grad = torch.randn(input_.shape, dtype=torch.bfloat16)
            grad_cpu = copy.deepcopy(grad).float()
            grad_mlu = grad.mlu()
            layer_norm = LayerNormModel(param[0], weight, bias)
            layer_norm_mlu = LayerNormModel(param[0], weight_mlu, bias_mlu)
            input_cpu = torch.nn.Parameter(input)
            input_mlu = torch.nn.Parameter(input_copy)
            out_cpu = layer_norm(input_cpu)
            out_cpu.backward(grad_cpu)
            out_mlu = layer_norm_mlu(input_mlu)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(),
                input_mlu.grad.cpu().float(),
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                layer_norm.layernorm.weight.grad.float(),
                layer_norm_mlu.layernorm.weight.grad.cpu().float(),
                0.003,
                use_MSE=True,
            )
            self.assertTensorsEqual(
                layer_norm.layernorm.bias.grad.float(),
                layer_norm_mlu.layernorm.bias.grad.cpu().float(),
                0.003,
                use_MSE=True,
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("58GB")
    def test_layer_norm_backward_large(self):
        # init: param | input: tensor
        gen_params = [((13725), (48, 4096, 13725))]
        dtype = torch.float
        for param in gen_params:
            weight = torch.randn(param[0], dtype=dtype, requires_grad=True)
            bias = torch.randn(param[0], dtype=dtype, requires_grad=True)
            weight_mlu = copy.deepcopy(weight)
            bias_mlu = copy.deepcopy(bias)
            input_ = torch.randn(param[1], dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_)
            grad_cpu = torch.randn(input_.shape, dtype=dtype)
            grad_mlu = copy.deepcopy(grad_cpu)
            layer_norm = LayerNormModel(param[0], weight, bias)
            layer_norm_mlu = LayerNormModel(param[0], weight_mlu, bias_mlu).mlu()
            out_cpu = layer_norm(input_)
            out_cpu.backward(grad_cpu)
            input_grad_cpu = copy.deepcopy(input_.grad)
            input_.grad.zero_()
            out_mlu = layer_norm_mlu(input_mlu.mlu())
            out_mlu.backward(grad_mlu.mlu())
            input_grad_mlu = copy.deepcopy(input_mlu.grad)
            input_mlu.grad.zero_()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_grad_cpu, input_grad_mlu.cpu(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
