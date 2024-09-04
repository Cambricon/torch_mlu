from __future__ import print_function

import sys
import logging
import os

import copy
import unittest
from itertools import product
import torch
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    mlufusion_on_and_off,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class PackedLSTM(torch.nn.Module):  # pylint: disable= W0223
    def __init__(self, input_size, hidden_size, num_layers, bias, bidirectional):
        super(PackedLSTM, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size, hidden_size, num_layers, bias=bias, bidirectional=bidirectional
        )

    def forward(self, input, input_lengths, ht, ct, assertFunc):
        x = torch.nn.utils.rnn.pack_padded_sequence(
            input, input_lengths, batch_first=True
        )

        if input.is_mlu:
            assertFunc(x.is_mlu, "Packed data is not on mlu device.")
        else:
            assertFunc(~(x.is_mlu), "Packed data is not on mlu device.")
        self.rnn.flatten_parameters()
        outputs, (hn, cn) = self.rnn(x, (ht, ct))

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hn, cn


class TestLstmOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    @mlufusion_on_and_off
    def test_packed_lstm_training(self):
        num_layers_v = [1, 2]
        input_size_v = [20]
        # hidden size value can't be to small. if hidden size is less than 50,
        # 3e-3 accuracy of gpu also is not satisfied.
        hidden_size_v = [128]
        batch_v = [16]
        seq_len_v = [15]
        bidirectional_v = [True, False]
        bias_v = [True, False]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        grad_contiguous_v = [True, False]
        amp_v = [False, True]
        mse_limited = 3e-3
        for (
            num_layers,
            input_size,
            hidden_size,
            batch,
            seq_len,
            bidirectional,
            bias,
            is_amp,
            func,
            is_grad_conti,
        ) in product(
            num_layers_v,
            input_size_v,
            hidden_size_v,
            batch_v,
            seq_len_v,
            bidirectional_v,
            bias_v,
            amp_v,
            func_list,
            grad_contiguous_v,
        ):
            cpu_rnn = PackedLSTM(
                input_size, hidden_size, num_layers, bias, bidirectional
            )
            mlu_rnn = copy.deepcopy(cpu_rnn).to("mlu")

            input = torch.randn(batch, seq_len, input_size)
            num_direction = 2 if bidirectional else 1
            h0 = torch.randn(num_layers * num_direction, batch, hidden_size)
            c0 = torch.randn(num_layers * num_direction, batch, hidden_size)

            input_length_list = [
                np.random.randint(seq_len / 2, seq_len) for i in range(batch)
            ]
            input_length_list.sort(reverse=True)
            input_length_list[0] = seq_len
            input_length = np.array(input_length_list)

            input_mlu = copy.deepcopy(input).to("mlu")
            h0_mlu = copy.deepcopy(h0).to("mlu")
            c0_mlu = copy.deepcopy(c0).to("mlu")
            h0_mlu.requires_grad = True
            c0_mlu.requires_grad = True
            input_mlu.requires_grad = True
            input.requires_grad = True
            h0.requires_grad = True
            c0.requires_grad = True

            # pytorch need h0, c0 need be contiguous.
            output_cpu, hn_cpu, cn_cpu = cpu_rnn(
                func(input), input_length, h0, c0, self.assertTrue
            )
            with torch.autocast("mlu", enabled=is_amp):
                output_mlu, hn_mlu, cn_mlu = mlu_rnn(
                    func(input_mlu), input_length, h0_mlu, c0_mlu, self.assertTrue
                )

            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(hn_cpu, hn_mlu.cpu(), 3e-3, use_MSE=True)
            self.assertTensorsEqual(cn_cpu, cn_mlu.cpu(), 3e-3, use_MSE=True)

            shape = [output_cpu.shape[0], output_cpu.shape[1], output_cpu.shape[2]]
            if is_grad_conti is False:
                shape[2] = 2 * shape[2]
            grad = torch.randn(shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")

            if is_grad_conti is False:
                output_cpu.backward(grad[..., ::2])
                output_mlu.backward(grad_mlu[..., ::2])
            else:
                output_cpu.backward(grad)
                output_mlu.backward(grad_mlu)

            self.assertTensorsEqual(
                input.grad, input_mlu.grad.cpu(), mse_limited, use_MSE=True
            )
            self.assertTensorsEqual(
                h0.grad, h0_mlu.grad.cpu(), mse_limited, use_MSE=True
            )
            self.assertTensorsEqual(
                c0.grad, c0_mlu.grad.cpu(), mse_limited, use_MSE=True
            )

            for w, w_mlu in zip(cpu_rnn.parameters(), mlu_rnn.parameters()):
                self.assertTensorsEqual(
                    w.grad.float(), w_mlu.grad.cpu().float(), mse_limited, use_MSE=True
                )


if __name__ == "__main__":
    unittest.main()
