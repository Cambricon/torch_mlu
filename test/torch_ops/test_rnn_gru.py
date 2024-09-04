from __future__ import print_function

import sys
import logging
import os
import unittest
from itertools import product
import torch
import torch.nn as nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestRNNGRUOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_rnn_gru_concat(self):
        def copy_rnn(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    x.data.copy_(y.data)

        def check_rnn_grads(rnn1, rnn2):
            for x_layer, y_layer in zip(rnn1.all_weights, rnn2.all_weights):
                for x, y in zip(x_layer, y_layer):
                    if y.dtype == torch.half:
                        self.assertTensorsEqual(
                            x.grad, y.grad.cpu().float(), 0.03, use_MSE=True
                        )
                    else:
                        self.assertEqual(x.grad, y.grad, atol=5e-5, rtol=0)

        input_size = 10
        hidden_size = 6
        num_layers = 2
        seq_length = 7
        batch = 6
        biass = [True, False]
        dtypes = [torch.float, torch.half]
        device = torch.device("mlu")
        funcs = [lambda x: x, self.to_non_dense]
        for dtype in dtypes:
            input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
            grad_output = torch.randn(seq_length, batch, hidden_size, dtype=dtype)
            hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
            grad_hy = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
            for module in (nn.RNN, nn.GRU):
                for bias, inp_f, hx_f, grado_f, gradh_f in product(
                    biass, funcs, funcs, funcs, funcs
                ):
                    rnn = module(input_size, hidden_size, num_layers, bias=bias).to(
                        dtype
                    )
                    if dtype == torch.half:
                        rnn = rnn.float()
                    rnn_device = module(
                        input_size, hidden_size, num_layers, bias=bias
                    ).to(device, dtype)
                    copy_rnn(rnn, rnn_device)

                    hx = hx_val.clone().requires_grad_(True)
                    hx_device = hx_val.clone().to(device).requires_grad_(True)

                    inp = input_val.clone().requires_grad_(True)
                    inp_cu = input_val.clone().to(device).requires_grad_(True)
                    output1, hy1 = rnn(
                        inp_f(inp.float() if dtype == torch.half else inp),
                        hx_f(hx.float() if dtype == torch.half else hx),
                    )
                    output2, hy2 = rnn_device(inp_f(inp_cu), hx_f(hx_device))
                    torch.autograd.backward(
                        [output1, hy1], [grado_f(grad_output), gradh_f(grad_hy)]
                    )
                    torch.autograd.backward(
                        [output2, hy2],
                        [grado_f(grad_output.to(device)), gradh_f(grad_hy.to(device))],
                    )
                    if dtype == torch.half:
                        self.assertTensorsEqual(
                            output1, output2.cpu().float(), 0.03, use_MSE=True
                        )
                        self.assertTensorsEqual(
                            hy1, hy2.cpu().float(), 0.03, use_MSE=True
                        )
                    else:
                        self.assertEqual(output1, output2)
                        self.assertEqual(hy1, hy2)

                    check_rnn_grads(rnn, rnn_device)
                    if dtype == torch.half:
                        self.assertTensorsEqual(
                            inp.grad, inp_cu.grad.cpu().float(), 0.03, use_MSE=True
                        )
                        self.assertTensorsEqual(
                            hx.grad, hx_device.grad.cpu().float(), 0.03, use_MSE=True
                        )
                    else:
                        self.assertEqual(inp.grad, inp_cu.grad)
                        self.assertEqual(hx.grad, hx_device.grad)


if __name__ == "__main__":
    unittest.main()
