from __future__ import print_function

import logging
import sys
import os
import unittest

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestAlias(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_alias_contiguous(self):
        dim = 64
        x = torch.randn((dim, dim), dtype=torch.float)
        out_cpu = x[:dim, :dim]
        x_mlu = x.mlu()
        out_mlu = x_mlu[:dim, :dim]
        out_mlu_ptr = out_mlu.data_ptr()
        x_mlu_ptr = x_mlu.data_ptr()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertEqual(out_mlu_ptr, x_mlu_ptr)

    # @unittest.skip("not test")
    @testinfo()
    def test_alias_channels_last(self):
        dim = 32
        shape = (64, 32, 128, 512)
        x = torch.randn(shape, dtype=torch.float)
        x = self.convert_to_channel_last(x)
        out_cpu = x[:dim, :dim, :dim, :dim]
        x_mlu = x.mlu()
        out_mlu = x_mlu[:dim, :dim, :dim, :dim]
        out_mlu_ptr = out_mlu.data_ptr()
        x_mlu_ptr = x_mlu.data_ptr()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertEqual(out_mlu_ptr, x_mlu_ptr)

    # @unittest.skip("not test")
    @testinfo()
    def test_alias_not_dense(self):
        dim = 32
        shape = (64, 32, 128, 512)
        x = torch.randn(shape, dtype=torch.float)[:, :, :, int(shape[-1] / 2)]
        x = self.convert_to_channel_last(x)
        out_cpu = x[:dim, :dim, :dim]
        x_mlu = x.mlu()
        out_mlu = x_mlu[:dim, :dim, :dim]
        out_mlu_ptr = out_mlu.data_ptr()
        x_mlu_ptr = x_mlu.data_ptr()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertEqual(out_mlu_ptr, x_mlu_ptr)


if __name__ == "__main__":
    unittest.main()
