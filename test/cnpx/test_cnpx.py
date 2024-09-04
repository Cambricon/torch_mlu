from __future__ import print_function

import os
import sys
import torch
import unittest
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase  # pylint: disable=C0411


class TestCNPX(TestCase):
    def test_cnpx(self):
        # Just making sure we can see the symbols
        torch.mlu.cnpx.range_push("foo")
        torch.mlu.cnpx.mark("bar")
        torch.mlu.cnpx.range_pop()
        range_handle = torch.mlu.cnpx.range_start("range_start")
        torch.mlu.cnpx.range_end(range_handle)


if __name__ == "__main__":
    unittest.main()
