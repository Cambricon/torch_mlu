from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
import random
from itertools import product

import torch
from torch import nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TEST_BFLOAT16,
    TestCase,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(2)


class TestDropoutOp(TestCase):
    def _test_dropout(
        self, cls, device, input, p, memory_format=torch.contiguous_format
    ):
        input = input.to(device).fill_(1 - p)

        module = cls(p)
        input_var = input.clone(memory_format=memory_format)
        input_var.requires_grad = True
        output = module(input_var)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))

        module = cls(p, True)
        input_var = input.clone(memory_format=memory_format)
        input_var.requires_grad = True
        output = module(input_var + 0)
        self.assertTrue(output.is_contiguous(memory_format=memory_format))
        output.backward(input)
        self.assertTrue(input_var.grad.is_contiguous(memory_format=memory_format))

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = cls(p, inplace).eval()
            self.assertEqual(input, module(input))

    # @unittest.skip("not test")
    @testinfo()
    def test_Fused_Dropout(self):
        input_fused = torch.randn(1000, requires_grad=True)
        grad_fused_mlu = torch.randn(1000).mlu()
        input_fused_mlu = input_fused.mlu()
        fused_res_mlu, fused_mask_mlu = torch._fused_dropout(input_fused_mlu, p=0.2)
        fused_res_mlu.backward(grad_fused_mlu)
        res_grad = input_fused.grad
        self.assertTrue(res_grad.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(
            fused_res_mlu.is_contiguous(memory_format=torch.contiguous_format)
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_native_dropout(self):
        input_native = torch.randn(1000, requires_grad=True)
        grad_native_mlu = torch.randn(1000).mlu()
        input_native_mlu = input_native.mlu()
        native_res_mlu, native_mask_mlu = torch.native_dropout(
            input_native_mlu, p=0.8, train=True
        )
        native_res_mlu.backward(grad_native_mlu)
        res_grad = input_native.grad
        self.assertTrue(res_grad.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(
            native_res_mlu.is_contiguous(memory_format=torch.contiguous_format)
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_Dropout(self):
        input = torch.Tensor(1000)
        self._test_dropout(nn.Dropout, "mlu", input, 0.2)
        input = torch.Tensor(1000).half()
        self._test_dropout(nn.Dropout, "mlu", input, 0.2)

    # @unittest.skip("not test")
    @testinfo()
    def test_Dropout2d(self):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        input = torch.Tensor(1000, b, w, h)
        self._test_dropout(nn.Dropout2d, "mlu", input, 0.2)
        input = torch.Tensor(1000, b, w, h).half()
        self._test_dropout(nn.Dropout2d, "mlu", input, 0.2)
        input = torch.Tensor(1000, b, w, h)
        self._test_dropout(nn.Dropout2d, "mlu", input, 0.6)
        input = torch.Tensor(1000, b, w, h)
        self._test_dropout(nn.Dropout2d, "mlu", input, 1)
        input = torch.Tensor(1000, b, w, h)
        self._test_dropout(nn.Dropout2d, "mlu", input, 0.8)
        input = torch.Tensor(1000, b, w, h)
        self._test_dropout(nn.Dropout2d, "mlu", input, 0.05)

    # @unittest.skip("not test")
    @testinfo()
    def test_Dropout2d_ChannelLast(self):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        input = torch.Tensor(1000, b, w, h)
        self._test_dropout(
            nn.Dropout2d, "mlu", input, 0.2, memory_format=torch.channels_last
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_Dropout3d(self):
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        d = random.randint(1, 5)
        input = torch.Tensor(1000, b, d, w, h)
        self._test_dropout(nn.Dropout3d, "mlu", input, 0.2)
        input = torch.Tensor(1000, b, d, w, h).half()
        self._test_dropout(nn.Dropout3d, "mlu", input, 0.2)

    # note: we only test the functions of dropout. we don`t test the performance of randomness
    #       because we haven't determine the test standard temporarily
    # @unittest.skip("not test")
    @testinfo()
    def test_alphadropout(self):
        m = nn.AlphaDropout(p=0.2)
        input = torch.randn(20, 16)
        output = m(input.to("mlu"))  # pylint: disable=W0612

    # @unittest.skip("not test")
    @testinfo()
    def test_featurealphadropout(self):
        m = nn.FeatureAlphaDropout(p=0.2)
        input = torch.ones(20, 16)
        output = m(input.to("mlu"))  # pylint: disable=W0612

    # @unittest.skip("not test")
    @testinfo()
    def test_dropout_set(self):
        m = nn.Dropout(p=0)
        input = torch.randn(10, 8)
        out_cpu = m(input)
        out_mlu = m(input.to("mlu"))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        m = nn.Dropout(p=1)
        input = torch.randn(10, 8)
        out_cpu = m(input)
        out_mlu = m(input.to("mlu"))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_native_dropout_bfloat16(self):
        input_fused = torch.testing.make_tensor(
            (2, 3, 24, 24), dtype=torch.bfloat16, device="cpu"
        )
        grad_fused = torch.testing.make_tensor(
            (2, 3, 24, 24), dtype=torch.bfloat16, device="cpu"
        )
        grad_fused_mlu = grad_fused.mlu()
        input_fused_cpu = torch.nn.Parameter(input_fused)
        input_fused_mlu = torch.nn.Parameter(input_fused.mlu())
        fused_res_cpu, fused_mask_cpu = torch.native_dropout(
            input_fused_cpu, p=0.2, train=True
        )
        fused_res_mlu, fused_mask_mlu = torch.native_dropout(
            input_fused_mlu, p=0.2, train=True
        )
        fused_res_cpu.backward(grad_fused)
        fused_res_mlu.backward(grad_fused_mlu)
        res_grad_cpu = input_fused_cpu.grad
        res_grad_mlu = input_fused_mlu.grad
        self.assertTrue(fused_res_cpu.dtype == fused_res_mlu.dtype)
        self.assertTrue(fused_mask_cpu.dtype == fused_mask_mlu.dtype)
        self.assertTrue(res_grad_cpu.dtype == res_grad_mlu.dtype)
        self.assertTrue(
            res_grad_cpu.size() == res_grad_mlu.size()
            and res_grad_cpu.stride() == res_grad_cpu.stride()
        )
        self.assertTrue(
            fused_res_cpu.size() == fused_res_mlu.size()
            and fused_res_cpu.stride() == fused_res_mlu.stride()
        )
        self.assertTrue(
            fused_mask_cpu.size() == fused_mask_mlu.size()
            and fused_mask_cpu.stride() == fused_mask_mlu.stride()
        )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_fused_dropout_bfloat16(self):
        input_fused = torch.randn(1000, dtype=torch.bfloat16, requires_grad=True)
        grad_fused_mlu = torch.randn(1000, dtype=torch.bfloat16).mlu()
        input_fused_mlu = input_fused.mlu()
        fused_res_mlu, fused_mask_mlu = torch._fused_dropout(input_fused_mlu, p=0.2)
        fused_res_mlu.backward(grad_fused_mlu)
        res_grad = input_fused.grad
        self.assertTrue(fused_res_mlu.dtype == torch.bfloat16)
        self.assertTrue(fused_mask_mlu.dtype == torch.uint8)
        self.assertTrue(res_grad.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(
            fused_res_mlu.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            fused_mask_mlu.is_contiguous(memory_format=torch.contiguous_format)
        )


if __name__ == "__main__":
    unittest.main()
