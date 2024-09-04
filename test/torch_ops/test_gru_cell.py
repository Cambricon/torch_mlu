from __future__ import print_function

import sys
import logging
import os

import copy
import unittest
from itertools import product
import torch
from torch import nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestGruCellOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_grucell_training(self):
        input_size_v = [832, 576]
        hidden_size_v = [512, 256, 0]
        batch_v = [128, 70, 0]
        bias_v = [True, False]
        dtype_v = [torch.float, torch.half]
        for input_size, hidden_size, batch, dtype, bias in product(
            input_size_v, hidden_size_v, batch_v, dtype_v, bias_v
        ):
            rnn = nn.GRUCell(input_size, hidden_size, bias=bias)
            input = nn.Parameter(torch.randn(batch, input_size))
            h0 = nn.Parameter(torch.randn(batch, hidden_size))
            hn = rnn(input, h0)
            grad = torch.randn(hn.shape)
            hn.backward(grad)
            input_grad = copy.deepcopy(input.grad)
            h0_grad = copy.deepcopy(h0.grad)
            w_grad = []
            for w in rnn.parameters():
                w_grad.append(copy.deepcopy(w.grad))
                w.grad.zero_()
            input.grad.zero_()
            h0.grad.zero_()

            rnn_mlu = rnn.to("mlu").to(dtype)
            input_mlu = input.to("mlu").to(dtype)
            h0_mlu = h0.to("mlu").to(dtype)
            hn_mlu = rnn_mlu(input_mlu, h0_mlu)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)

            grad_mlu = grad.to("mlu").to(dtype)
            hn_mlu.backward(grad_mlu)
            input_mlu_grad = copy.deepcopy(input.grad)
            h0_mlu_grad = copy.deepcopy(h0.grad)
            w_grad_mlu = []
            for w in rnn.parameters():
                w_grad_mlu.append(copy.deepcopy(w.grad))
                w.grad.zero_()
            input.grad.zero_()
            h0.grad.zero_()
            self.assertTensorsEqual(
                input_grad, input_mlu_grad.cpu(), 3e-3, use_MSE=True
            )
            self.assertTensorsEqual(h0_grad, h0_mlu_grad.cpu(), 3e-3, use_MSE=True)

            for w, w_mlu in zip(w_grad, w_grad_mlu):
                self.assertTensorsEqual(w, w_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_grucell_training_not_contiguous(self):
        input_size_v = [1024, 384]
        hidden_size_v = [512, 256, 0]
        batch_v = [128, 94, 0]
        bias_v = [True, False]
        dtype_v = [torch.float, torch.half]
        for input_size, hidden_size, batch, dtype, bias in product(
            input_size_v, hidden_size_v, batch_v, dtype_v, bias_v
        ):
            rnn = nn.GRUCell(input_size, hidden_size, bias=bias)
            input_size *= 2
            input = nn.Parameter(torch.randn(batch, input_size))
            h0 = nn.Parameter(torch.randn(batch, hidden_size * 2))
            hn = rnn(input[:, ::2], h0[:, ::2])
            shape = (hn.shape[0], hn.shape[1] * 2)
            grad = torch.randn(shape)
            hn.backward(grad[:, ::2])
            input_grad = copy.deepcopy(input.grad)
            h0_grad = copy.deepcopy(h0.grad)
            w_grad = []
            for w in rnn.parameters():
                w_grad.append(copy.deepcopy(w.grad))
                w.grad.zero_()
            input.grad.zero_()
            h0.grad.zero_()

            rnn_mlu = rnn.to("mlu").to(dtype)
            input_mlu = input.to("mlu").to(dtype)[:, ::2]
            h0_mlu = h0.to("mlu").to(dtype)[:, ::2]
            hn_mlu = rnn_mlu(input_mlu, h0_mlu)
            grad_mlu = grad.to("mlu").to(dtype)
            hn_mlu.backward(grad_mlu[:, ::2])
            input_mlu_grad = copy.deepcopy(input.grad)
            h0_mlu_grad = copy.deepcopy(h0.grad)
            w_grad_mlu = []
            for w in rnn.parameters():
                w_grad_mlu.append(copy.deepcopy(w.grad))
                w.grad.zero_()
            input.grad.zero_()
            h0.grad.zero_()
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(
                input_grad, input_mlu_grad.cpu(), 3e-3, use_MSE=True
            )
            self.assertTensorsEqual(h0_grad, h0_mlu_grad.cpu(), 3e-3, use_MSE=True)

            for w, w_mlu in zip(w_grad, w_grad_mlu):
                self.assertTensorsEqual(w, w_mlu.cpu().float(), 3e-3, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("73GB")
    def test_grucell_training_large(self):
        input_size = 1024 * 1024
        hidden_size = 1024
        batch = 4 * 1025
        bias = False
        dtype = torch.float
        rnn = nn.GRUCell(input_size, hidden_size, bias=bias)
        input = nn.Parameter(torch.randn(batch, input_size))
        h0 = nn.Parameter(torch.randn(batch, hidden_size))
        hn = rnn(input, h0)
        grad = torch.randn(hn.shape)
        hn.backward(grad)
        input_grad = copy.deepcopy(input.grad)
        h0_grad = copy.deepcopy(h0.grad)
        w_grad = []
        for w in rnn.parameters():
            w_grad.append(copy.deepcopy(w.grad))
            w.grad.zero_()
        input.grad.zero_()
        h0.grad.zero_()

        rnn_mlu = rnn.to("mlu").to(dtype)
        input_mlu = input.to("mlu").to(dtype)
        h0_mlu = h0.to("mlu").to(dtype)
        hn_mlu = rnn_mlu(input_mlu, h0_mlu)
        self.assertTensorsEqual(hn, hn_mlu.cpu().float(), 3e-3, use_MSE=True)

        grad_mlu = grad.to("mlu").to(dtype)
        hn_mlu.backward(grad_mlu)
        input_mlu_grad = copy.deepcopy(input.grad)
        h0_mlu_grad = copy.deepcopy(h0.grad)
        w_grad_mlu = []
        for w in rnn.parameters():
            w_grad_mlu.append(copy.deepcopy(w.grad))
            w.grad.zero_()
        input.grad.zero_()
        h0.grad.zero_()
        self.assertTensorsEqual(
            input_grad, input_mlu_grad.cpu().float(), 3e-3, use_MSE=True
        )
        self.assertTensorsEqual(h0_grad, h0_mlu_grad.cpu().float(), 3e-3, use_MSE=True)

        for w, w_mlu in zip(w_grad, w_grad_mlu):
            self.assertTensorsEqual(w, w_mlu.cpu().float(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    run_tests()
