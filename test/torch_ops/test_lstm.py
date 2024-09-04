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
    TestCase,
    mlufusion_on_and_off,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestLstmOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training_basic(self):
        num_layers = 1
        input_size_v = [5, 9]
        hidden_size_v = [4, 7]
        batch_v = [3, 6]
        seq_len_v = [2, 9]
        bidirectional_v = [False, True]
        bias_v = [True, False]
        proj_size_v = [0, 2]
        for (
            input_size,
            hidden_size,
            batch,
            seq_len,
            bidirectional,
            bias,
            proj_size,
        ) in product(
            input_size_v,
            hidden_size_v,
            batch_v,
            seq_len_v,
            bidirectional_v,
            bias_v,
            proj_size_v,
        ):
            rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                bidirectional=bidirectional,
                proj_size=proj_size,
            )
            input = torch.randn(seq_len, batch, input_size)
            output, (hn, cn) = rnn(input)

            rnn_mlu = rnn.to("mlu")
            input_mlu = input.to("mlu")
            input_mlu.requires_grad = True
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu)
            self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training_basic_batchfirst(self):
        num_layers = 1
        input_size_v = [5, 9]
        hidden_size_v = [4, 7]
        batch_v = [3, 6]
        seq_len_v = [2, 9]
        bidirectional_v = [True, False]
        bias_v = [True, False]
        proj_size_v = [0, 3]
        for (
            input_size,
            hidden_size,
            batch,
            seq_len,
            bidirectional,
            bias,
            proj_size,
        ) in product(
            input_size_v,
            hidden_size_v,
            batch_v,
            seq_len_v,
            bidirectional_v,
            bias_v,
            proj_size_v,
        ):
            rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                bidirectional=bidirectional,
                batch_first=True,
                proj_size=proj_size,
            )
            input = torch.randn(batch, seq_len, input_size)
            output, (hn, cn) = rnn(input)

            rnn_mlu = rnn.to("mlu")
            input_mlu = input.to("mlu")
            input_mlu.requires_grad = True
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu)
            self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_inference_basic(self):
        num_layers = 1
        input_size_v = [5, 9]
        hidden_size_v = [4, 7]
        batch_v = [3, 6]
        seq_len_v = [2, 9]
        bidirectional_v = [True, False]
        bias_v = [True, False]
        proj_size_v = [0, 3]
        for (
            input_size,
            hidden_size,
            batch,
            seq_len,
            bidirectional,
            bias,
            proj_size,
        ) in product(
            input_size_v,
            hidden_size_v,
            batch_v,
            seq_len_v,
            bidirectional_v,
            bias_v,
            proj_size_v,
        ):
            rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                bidirectional=bidirectional,
                proj_size=proj_size,
            )
            input = torch.randn(seq_len, batch, input_size)
            rnn.eval()
            output, (hn, cn) = rnn(input)

            rnn_mlu = rnn.to("mlu")
            input_mlu = input.to("mlu")
            input_mlu.requires_grad = True
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu)
            self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_inference_batchfist(self):
        num_layers = 1
        input_size_v = [5, 9]
        hidden_size_v = [4, 7]
        batch_v = [3, 6]
        seq_len_v = [2, 9]
        bidirectional_v = [True, False]
        bias_v = [True, False]
        proj_size_v = [0, 3]
        for (
            input_size,
            hidden_size,
            batch,
            seq_len,
            bidirectional,
            bias,
            proj_size,
        ) in product(
            input_size_v,
            hidden_size_v,
            batch_v,
            seq_len_v,
            bidirectional_v,
            bias_v,
            proj_size_v,
        ):
            rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                bidirectional=bidirectional,
                batch_first=True,
                proj_size=proj_size,
            )
            input = torch.randn(batch, seq_len, input_size)
            rnn.eval()
            output, (hn, cn) = rnn(input)

            rnn_mlu = rnn.to("mlu")
            input_mlu = input.to("mlu")
            input_mlu.requires_grad = True
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu)
            self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training_dtype(self):
        for dtype in [torch.half, torch.double]:
            rnn = nn.LSTM(10, 20, 1, bias=False, bidirectional=True)
            input = torch.randn(5, 3, 10)
            h0 = torch.randn(2, 3, 20)
            c0 = torch.randn(2, 3, 20)

            rnn_mlu = copy.deepcopy(rnn).to("mlu")
            rnn_mlu.to(dtype)
            input_mlu = copy.deepcopy(input).to("mlu").to(dtype)
            h0_mlu = copy.deepcopy(h0).to("mlu").to(dtype)
            c0_mlu = copy.deepcopy(c0).to("mlu").to(dtype)
            h0_mlu.requires_grad = True
            c0_mlu.requires_grad = True
            input_mlu.requires_grad = True
            input.requires_grad = True
            h0.requires_grad = True
            c0.requires_grad = True
            output, (hn, cn) = rnn(input, (h0, c0))
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu, (h0_mlu, c0_mlu))
            self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

            grad = torch.randn(out_mlu.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")

            output.backward(grad)
            out_mlu.backward(grad_mlu)

            self.assertTensorsEqual(
                input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True
            )
            self.assertTensorsEqual(h0.grad, h0_mlu.grad.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(c0.grad, c0_mlu.grad.cpu(), 3e-3, use_MSE=True)

            for w, w_mlu in zip(rnn.parameters(), rnn_mlu.parameters()):
                self.assertTensorsEqual(w.grad, w_mlu.grad.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training_uncontiguous(self):
        rnn = nn.LSTM(10, 20, 1, bias=False, bidirectional=True)
        input = torch.randn(10, 3, 5)
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)

        rnn_mlu = copy.deepcopy(rnn).to("mlu")
        input_mlu = copy.deepcopy(input).to("mlu")
        input = input.permute([2, 1, 0])
        input_mlu = input_mlu.permute([2, 1, 0])
        h0_mlu = copy.deepcopy(h0).to("mlu")
        c0_mlu = copy.deepcopy(c0).to("mlu")
        h0_mlu.requires_grad = True
        c0_mlu.requires_grad = True
        input_mlu.requires_grad = True
        input.requires_grad = True
        h0.requires_grad = True
        c0.requires_grad = True
        output, (hn, cn) = rnn(input, (h0, c0))
        out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu, (h0_mlu, c0_mlu))
        self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

        grad = torch.randn(out_mlu.shape)
        grad_mlu = copy.deepcopy(grad).to("mlu")

        output.backward(grad)
        out_mlu.backward(grad_mlu)

        self.assertTensorsEqual(input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(h0.grad, h0_mlu.grad.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(c0.grad, c0_mlu.grad.cpu(), 3e-3, use_MSE=True)

        for w, w_mlu in zip(rnn.parameters(), rnn_mlu.parameters()):
            self.assertTensorsEqual(w.grad, w_mlu.grad.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training_batchfirst(self):
        rnn = nn.LSTM(10, 20, 1, bias=False, bidirectional=True, batch_first=True)
        input = torch.randn(5, 3, 10).transpose(0, 1).contiguous()
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)

        rnn_mlu = copy.deepcopy(rnn).to("mlu")
        input_mlu = copy.deepcopy(input).to("mlu")
        h0_mlu = copy.deepcopy(h0).to("mlu")
        c0_mlu = copy.deepcopy(c0).to("mlu")
        h0_mlu.requires_grad = True
        c0_mlu.requires_grad = True
        input_mlu.requires_grad = True
        input.requires_grad = True
        h0.requires_grad = True
        c0.requires_grad = True
        output, (hn, cn) = rnn(input, (h0, c0))
        out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu, (h0_mlu, c0_mlu))
        self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

        grad = torch.randn(5, 3, 40).transpose(0, 1).contiguous()
        grad_mlu = copy.deepcopy(grad).to("mlu")

        output.backward(grad)
        out_mlu.backward(grad_mlu)

        self.assertTensorsEqual(input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(h0.grad, h0_mlu.grad.cpu(), 3e-3, use_MSE=True)
        self.assertTensorsEqual(c0.grad, c0_mlu.grad.cpu(), 3e-3, use_MSE=True)

        for w, w_mlu in zip(rnn.parameters(), rnn_mlu.parameters()):
            self.assertTensorsEqual(w.grad, w_mlu.grad.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training(self):
        num_layers = 1
        input_size_v = [3, 5, 8]
        hidden_size_v = [3, 4, 7]
        batch_v = [0, 3, 5, 8]
        seq_len_v = [3, 6, 9]
        bidirectional_v = [True, False]
        bias_v = [True, False]
        proj_size_v = [0, 2]
        for (
            input_size,
            hidden_size,
            batch,
            seq_len,
            bidirectional,
            bias,
            proj_size,
        ) in product(
            input_size_v,
            hidden_size_v,
            batch_v,
            seq_len_v,
            bidirectional_v,
            bias_v,
            proj_size_v,
        ):
            rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                bidirectional=bidirectional,
                proj_size=proj_size,
            )
            input = torch.randn(seq_len, batch, input_size)

            h0 = torch.randn(
                (int)(bidirectional) + 1,
                batch,
                proj_size if proj_size > 0 else hidden_size,
            )
            c0 = torch.randn((int)(bidirectional) + 1, batch, hidden_size)

            rnn_mlu = copy.deepcopy(rnn).to("mlu")
            input_mlu = copy.deepcopy(input).to("mlu")
            h0_mlu = copy.deepcopy(h0).to("mlu")
            c0_mlu = copy.deepcopy(c0).to("mlu")
            h0_mlu.requires_grad = True
            c0_mlu.requires_grad = True
            input_mlu.requires_grad = True
            input.requires_grad = True
            h0.requires_grad = True
            c0.requires_grad = True
            output, (hn, cn) = rnn(input, (h0, c0))
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu, (h0_mlu, c0_mlu))
            self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

            grad = torch.randn(out_mlu.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")

            output.backward(grad)
            out_mlu.backward(grad_mlu)

            self.assertTensorsEqual(
                input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True
            )
            self.assertTensorsEqual(h0.grad, h0_mlu.grad.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(c0.grad, c0_mlu.grad.cpu(), 3e-3, use_MSE=True)

            for w, w_mlu in zip(rnn.parameters(), rnn_mlu.parameters()):
                self.assertTensorsEqual(w.grad, w_mlu.grad.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training_numlayers(self):
        # TODO(shangang): numlayers increase will decrease op percision.  # pylint: disable=W0511
        num_layers = 2
        input_size_v = [3, 5, 8]
        hidden_size_v = [3, 4, 7]
        batch_v = [3, 5, 8]
        seq_len_v = [3, 6, 9]
        bidirectional_v = [True, False]
        bias_v = [True, False]
        proj_size_v = [0, 2]
        # TODO(shangang): numlayers increase will decrease op percision.  # pylint: disable=W0511
        mse_limited = 3e-3
        for (
            input_size,
            hidden_size,
            batch,
            seq_len,
            bidirectional,
            bias,
            proj_size,
        ) in product(
            input_size_v,
            hidden_size_v,
            batch_v,
            seq_len_v,
            bidirectional_v,
            bias_v,
            proj_size_v,
        ):
            rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                bidirectional=bidirectional,
                proj_size=proj_size,
            )
            input = torch.randn(seq_len, batch, input_size)
            h0 = torch.randn(
                ((int)(bidirectional) + 1) * num_layers,
                batch,
                proj_size if proj_size > 0 else hidden_size,
            )
            c0 = torch.randn(
                ((int)(bidirectional) + 1) * num_layers, batch, hidden_size
            )

            rnn_mlu = copy.deepcopy(rnn).to("mlu")
            input_mlu = copy.deepcopy(input).to("mlu")
            h0_mlu = copy.deepcopy(h0).to("mlu")
            c0_mlu = copy.deepcopy(c0).to("mlu")
            h0_mlu.requires_grad = True
            c0_mlu.requires_grad = True
            input_mlu.requires_grad = True
            input.requires_grad = True
            h0.requires_grad = True
            c0.requires_grad = True
            output, (hn, cn) = rnn(input, (h0, c0))
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu, (h0_mlu, c0_mlu))
            self.assertTensorsEqual(output, out_mlu.cpu(), mse_limited, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), mse_limited, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), mse_limited, use_MSE=True)

            grad = torch.randn(out_mlu.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")

            output.backward(grad)
            out_mlu.backward(grad_mlu)

            self.assertTensorsEqual(
                input.grad, input_mlu.grad.cpu(), mse_limited, use_MSE=True
            )
            self.assertTensorsEqual(
                h0.grad, h0_mlu.grad.cpu(), mse_limited, use_MSE=True
            )
            self.assertTensorsEqual(
                c0.grad, c0_mlu.grad.cpu(), mse_limited, use_MSE=True
            )

            for w, w_mlu in zip(rnn.parameters(), rnn_mlu.parameters()):
                self.assertTensorsEqual(
                    w.grad, w_mlu.grad.cpu(), mse_limited, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_lstm_exception(self):
        orig_flg = torch.backends.mlufusion.enabled
        torch.backends.mlufusion.set_flags(True)
        rnn = nn.LSTM(10, 20, 1, bias=True, bidirectional=True)
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(20, 2, 3)
        c0 = torch.randn(20, 2, 3)
        rnn_mlu = rnn.to("mlu")
        input_mlu = input.to("mlu")
        h0_mlu = h0.to("mlu").permute([1, 2, 0])
        c0_mlu = c0.to("mlu").permute([1, 2, 0])
        ref_msg = "rnn: hx is not contiguous"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(
                input_mlu, (h0_mlu, c0_mlu)
            )  # pylint: disable=W0612

        ref_msg = "rnn: cx is not contiguous"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(
                input_mlu, (h0_mlu.contiguous(), c0_mlu)
            )

        ref_msg = (
            r"Input and hidden tensors are not at the same device,"
            + r" found input tensor at cpu and hidden tensor at mlu:0"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(
                input_mlu.cpu(), (h0_mlu.contiguous(), c0_mlu)
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    @mlufusion_on_and_off
    def test_lstm_training_bfloat16(self):
        for dtype in [
            torch.bfloat16,
        ]:
            rnn = nn.LSTM(10, 20, 1, bias=False, bidirectional=True)
            input = torch.randn(5, 3, 10)
            h0 = torch.randn(2, 3, 20)
            c0 = torch.randn(2, 3, 20)

            rnn_mlu = copy.deepcopy(rnn).to("mlu")
            rnn_mlu.to(dtype)
            input_mlu = copy.deepcopy(input).to("mlu").to(dtype)
            h0_mlu = copy.deepcopy(h0).to("mlu").to(dtype)
            c0_mlu = copy.deepcopy(c0).to("mlu").to(dtype)
            h0_mlu.requires_grad = True
            c0_mlu.requires_grad = True
            input_mlu.requires_grad = True
            input.requires_grad = True
            h0.requires_grad = True
            c0.requires_grad = True
            output, (hn, cn) = rnn(input, (h0, c0))
            out_mlu, (hn_mlu, cn_mlu) = rnn_mlu(input_mlu, (h0_mlu, c0_mlu))
            self.assertTensorsEqual(output, out_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn, cn_mlu.cpu(), 3e-3, use_MSE=True)

            grad = torch.randn(out_mlu.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")

            output.backward(grad)
            out_mlu.backward(grad_mlu)

            self.assertTensorsEqual(
                input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True
            )
            self.assertTensorsEqual(h0.grad, h0_mlu.grad.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(c0.grad, c0_mlu.grad.cpu(), 3e-3, use_MSE=True)

            for w, w_mlu in zip(rnn.parameters(), rnn_mlu.parameters()):
                self.assertTensorsEqual(w.grad, w_mlu.grad.cpu(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
