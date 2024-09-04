from __future__ import print_function

import sys
import logging
import copy
import os
import unittest
from itertools import product
import torch
from torch.nn import GroupNorm

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
    run_tests,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

NC = [(0, 1), (1, 2), (2, 9), (4, 16), (3, 64), (15, 8), (7, 27)]
G = [1, 1, 3, 16, 4, 8, 27]

HxW = [
    (),
    (1,),
    (2, 0),
    (1, 1),
    (2, 7),
    (3, 5, 4, 2),
    (1, 1, 1, 1, 1, 1),
    (5, 8, 4, 1, 7, 3),
]

affines = [True, False]
dtypes = [torch.float, torch.half, torch.double]


def shape_list():
    shapes = []
    groups = []
    nd_shapes = []
    for i, sf in enumerate(NC):
        group = G[i]
        for ef in HxW:
            shape = sf + ef
            shapes.append(shape)
            groups.append(group)
            nd_shape = shape + (2,)
            nd_shapes.append(nd_shape)
    return shapes, groups, nd_shapes


shapes, groups, nd_shapes = shape_list()


class TestGroupNormOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_contiguous(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group in zip(shapes, groups):
                layer = GroupNorm(group, shape[1], affine=affine)
                x = torch.randn(shape).to(dtype)
                x_mlu = x.to("mlu")
                out_cpu = layer(x.to(torch.float))
                layer = layer.to("mlu").to(dtype)
                out_mlu = layer(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_channel_last(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group in zip(shapes, groups):
                layer = GroupNorm(group, shape[1], affine=affine)
                x = self.convert_to_channel_last(torch.randn(shape).to(dtype))
                x_mlu = x.to("mlu")
                out_cpu = layer(x.to(torch.float))
                layer = layer.to("mlu").to(dtype)
                out_mlu = layer(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_not_dense(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group, nd_shape in zip(shapes, groups, nd_shapes):
                layer = GroupNorm(group, shape[1], affine=affine)
                x_nd = torch.randn(nd_shape).to(dtype)
                x = x_nd[..., 1]
                x_mlu = x.to("mlu")
                out_cpu = layer(x.to(torch.float))
                layer = layer.to("mlu").to(dtype)
                out_mlu = layer(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.03, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_backward(self):
        for affine, dtype in product(affines, dtypes):
            for shape, group, nd_shape in zip(shapes, groups, nd_shapes):
                if dtype == torch.half:
                    dtype = torch.float
                # (FIXME): group_norm_backward is insufficient accuracy in some dtype and some shape
                er = 0.05
                layer = GroupNorm(group, shape[1], affine=affine)
                x_nd = torch.randn(nd_shape, dtype=dtype, requires_grad=True)
                x = x_nd[..., 1]
                x_mlu = x.to("mlu")
                out_cpu = layer(x.to(torch.float))
                grad = torch.randn(nd_shape, dtype=dtype)
                grad_cpu = grad.to(out_cpu.dtype)[..., 1].view(out_cpu.shape)
                out_cpu.backward(grad_cpu)
                x_grad_cpu = copy.deepcopy(x_nd.grad)
                x_nd.grad.zero_()
                if affine:
                    gamma_grad_cpu = copy.deepcopy(layer.weight.grad)
                    beta_grad_cpu = copy.deepcopy(layer.bias.grad)
                    layer.weight.grad.zero_()
                    layer.bias.grad.zero_()
                layer = layer.to("mlu").to(dtype)
                out_mlu = layer(x_mlu)
                grad_mlu = grad.to("mlu").to(out_mlu.dtype)[..., 1].view(out_mlu.shape)
                out_mlu.backward(grad_mlu)
                x_grad_mlu = copy.deepcopy(x_nd.grad)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), er, use_MSE=True
                )
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x_grad_mlu.cpu().float(), er, use_MSE=True
                )
                if affine:
                    gamma_grad_mlu = copy.deepcopy(layer.weight.grad)
                    beta_grad_mlu = copy.deepcopy(layer.bias.grad)
                    self.assertTensorsEqual(
                        gamma_grad_cpu, gamma_grad_mlu.cpu().float(), er, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        beta_grad_cpu, beta_grad_mlu.cpu().float(), er, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_group_norm_exceptions(self):
        shape = (3, 9, 3, 3)
        x = torch.randn(shape)
        x_mlu = x.to("mlu")
        layer = GroupNorm(3, 6)
        layer = layer.to("mlu")
        msg = (
            "Expected weight to be a vector of size equal to the number of channels in "
            + "input, but got weight of shape [6] and input of shape [3, 9, 3, 3]"
        )
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            layer(x_mlu)
        self.assertEqual(err_msg_mlu.exception.args[0], msg)

        layer = GroupNorm(3, 9)
        layer = layer.to("mlu")
        x_mlu = x_mlu.to(torch.int)
        msg = "GroupNorm only support float, bfloat16, half and double type inputs, but got dtype: Int"
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            layer(x_mlu)
        self.assertEqual(err_msg_mlu.exception.args[0], msg)

        layer = GroupNorm(3, 9)
        layer = layer.to("mlu")
        x_mlu = x_mlu.to(torch.half)
        msg = (
            "GroupNorm only support same dtypes of input, weight and bias, but got "
            + "input dtype: Half weight dtype: Float bias dtype: Float"
        )
        with self.assertRaises(RuntimeError) as err_msg_mlu:
            layer(x_mlu)
        self.assertEqual(err_msg_mlu.exception.args[0], msg)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_group_norm_bfloat16_forward(self):
        affine = False
        dtype = torch.bfloat16
        for shape, group, nd_shape in zip(shapes, groups, nd_shapes):
            # (FIXME): group_norm_backward is insufficient accuracy in some dtype and some shape
            er = 0.05
            layer = GroupNorm(group, shape[1], affine=affine)
            x_nd = torch.randn(nd_shape, dtype=dtype, requires_grad=True)
            x = x_nd[..., 1]
            x_mlu = x.to("mlu")
            out_cpu = layer(x.to(torch.float))
            layer = layer.to("mlu").to(dtype)
            out_mlu = layer(x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), er, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_group_norm_bfloat16_backward(self):
        torch.manual_seed(12)
        affine = False
        dtype = torch.bfloat16
        shape, group, nd_shape = (0, 1, 2, 7), 1, (0, 1, 2, 7, 2)
        # (FIXME): group_norm_backward has insufficient accuracy for dtype bf16
        er = 0.05
        layer = GroupNorm(group, shape[1], affine=affine)
        x_nd = torch.randn(nd_shape, dtype=dtype, requires_grad=True)
        x = x_nd[..., 1]
        x_mlu = x.to("mlu")
        out_cpu = layer(x.to(torch.float))
        grad = torch.randn(nd_shape, dtype=dtype)
        grad_cpu = grad.to(out_cpu.dtype)[..., 1].view(out_cpu.shape)
        out_cpu.backward(grad_cpu)
        x_grad_cpu = copy.deepcopy(x_nd.grad)
        x_nd.grad.zero_()
        layer = layer.to("mlu").to(dtype)
        out_mlu = layer(x_mlu)
        grad_mlu = grad.to("mlu").to(out_mlu.dtype)[..., 1].view(out_mlu.shape)
        out_mlu.backward(grad_mlu)
        x_grad_mlu = copy.deepcopy(x_nd.grad)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), er, use_MSE=True)
        self.assertTensorsEqual(
            x_grad_cpu.float(), x_grad_mlu.cpu().float(), er, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("61GB")
    def test_group_norm_backward_large_half(self):
        device = "mlu"
        dtype = torch.half
        shape = (4, 1025, 1024, 1024)
        g = 1025
        grad = True
        x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
        x.requires_grad_(grad)
        b = shape[0]
        c = shape[1]

        # test that GN normalizes to mean 0 and stddev 1
        gn = torch.nn.GroupNorm(g, c, eps=0).to(device, dtype)
        gn.weight.data.fill_(1)
        gn.bias.data.fill_(0)
        output = gn(x)
        out_reshaped = output.view(b, g, -1)
        mean = out_reshaped.mean(-1)
        var = out_reshaped.cpu().var(-1, unbiased=False)
        self.assertEqual(torch.abs(mean).mean(), 0, atol=0.05, rtol=0)
        self.assertEqual(torch.abs(var).mean(), 1, atol=0.05, rtol=0)

        output.backward(torch.randn_like(output))
        if output.is_mlu:
            torch.mlu.synchronize()

        # test that GN applies weight and bias correctly
        scale = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
        bias = torch.empty(c, device=device, dtype=dtype).uniform_(0.2, 2)
        gn.weight.data.copy_(scale)
        gn.bias.data.copy_(bias)
        output = gn(x)
        out_reshaped = output.view(b, c, -1)
        out_normed = (out_reshaped - bias.view(c, 1)) / scale.view(c, 1)
        out_normed_reshaped = out_normed.view(b, g, -1)
        mean = out_normed_reshaped.mean(-1)
        var = out_normed_reshaped.cpu().var(-1, unbiased=False)
        self.assertEqual(torch.abs(mean).mean(), 0, atol=0.05, rtol=0)
        self.assertEqual(torch.abs(var).mean(), 1, atol=0.05, rtol=0)


if __name__ == "__main__":
    run_tests()
