from __future__ import print_function

import sys
import logging
import os
import unittest
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestRNNUtilsOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_pad_sequence(self):
        a = torch.randn(25, 300)
        a_mlu = a.to("mlu")
        b = torch.randn(22, 300)
        b_mlu = b.to("mlu")
        c = torch.randn(15, 300)
        c_mlu = c.to("mlu")
        batch_first_list = [True, False]
        for batch_first in batch_first_list:
            output_cpu = pad_sequence([a, b, c], batch_first=batch_first)
            output_mlu = pad_sequence([a_mlu, b_mlu, c_mlu], batch_first=batch_first)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pack_padded_sequence(self):
        seq = torch.randn(128, 86, 512)
        lens = list(np.random.randint(43, 86, size=(128)))

        packed_cpu = pack_padded_sequence(
            seq, lens, batch_first=True, enforce_sorted=False
        )
        packed_mlu = pack_padded_sequence(
            seq.to("mlu"), lens, batch_first=True, enforce_sorted=False
        )
        self.assertTensorsEqual(
            packed_cpu.data, packed_mlu.data.cpu(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            packed_cpu.batch_sizes, packed_mlu.batch_sizes, 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            packed_cpu.sorted_indices,
            packed_mlu.sorted_indices.cpu(),
            0.0,
            use_MSE=True,
        )
        self.assertTensorsEqual(
            packed_cpu.unsorted_indices,
            packed_mlu.unsorted_indices.cpu(),
            0.0,
            use_MSE=True,
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_pad_packed_sequence(self):
        seq = torch.randn(128, 86, 512)
        lens = list(np.random.randint(43, 86, size=(128)))

        packed_cpu = pack_padded_sequence(
            seq, lens, batch_first=True, enforce_sorted=False
        )
        packed_mlu = pack_padded_sequence(
            seq.to("mlu"), lens, batch_first=True, enforce_sorted=False
        )

        seq_unpacked_cpu, lens_unpacked_cpu = pad_packed_sequence(
            packed_cpu, batch_first=True
        )
        seq_unpacked_mlu, lens_unpacked_mlu = pad_packed_sequence(
            packed_mlu, batch_first=True
        )

        self.assertTensorsEqual(
            seq_unpacked_cpu, seq_unpacked_mlu.cpu(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            lens_unpacked_cpu, lens_unpacked_mlu.cpu(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
