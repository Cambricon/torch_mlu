from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


class TestEmptyStrided(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_empty_strided(self):
        shape_stride_list = [((2, 3), (1, 2)), ((6, 7, 8), (1, 4, 2))]
        for shape, stride in shape_stride_list:
            x = torch.empty_strided(shape, stride, device="mlu")
            x_cpu = torch.empty_strided(shape, stride)
            self.assertEqual(x_cpu.size(), x.size())
            self.assertEqual(x_cpu.stride(), x.stride())


if __name__ == "__main__":
    unittest.main()
