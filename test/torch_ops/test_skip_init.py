import sys
import logging
import os
import unittest
import numpy as np
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase, run_tests

logging.basicConfig(level=logging.DEBUG)


class TestSkipInitMLU(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_skip_init(self):
        m1 = torch.nn.utils.skip_init(torch.nn.Linear, 5, 1, device="mlu")
        assert m1.weight.shape == (1, 5)
        m2 = torch.nn.utils.skip_init(
            torch.nn.Linear, in_features=6, out_features=1, device="mlu"
        )
        assert m2.weight.shape == (1, 6)
        m3 = torch.nn.utils.skip_init(
            torch.nn.Conv1d, kernel_size=3, in_channels=1, out_channels=1, device="mlu"
        )
        assert m3.weight.shape == (1, 1, 3)
        m4 = torch.nn.utils.skip_init(
            torch.nn.Conv2d, kernel_size=10, in_channels=3, out_channels=3, device="mlu"
        )
        assert m4.weight.shape == (3, 3, 10, 10)
        m5 = torch.nn.utils.skip_init(
            torch.nn.Conv3d, kernel_size=6, in_channels=4, out_channels=4, device="mlu"
        )
        assert m5.weight.shape == (4, 4, 6, 6, 6)
        m6 = torch.nn.utils.skip_init(
            torch.nn.Embedding, num_embeddings=3, embedding_dim=4, device="mlu"
        )
        assert m6.weight.shape == (3, 4)


if __name__ == "__main__":
    run_tests()
