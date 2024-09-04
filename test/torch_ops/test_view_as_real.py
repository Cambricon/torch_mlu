from __future__ import print_function

import unittest
import logging
from itertools import product
from contextlib import contextmanager
import sys
import os
import librosa
import numpy as np
import torch


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestViewAsRealOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_view_as_real(self):
        def fn(dtype, contiguous_input=True):
            t = torch.randn(3, 4, dtype=dtype).to("mlu")
            if not contiguous_input:
                t = t.transpose(0, 1)
            res = torch.view_as_real(t)
            self.assertTensorsEqual(res[:, :, 0].cpu(), t.real.cpu(), 0)
            self.assertTensorsEqual(res[:, :, 1].cpu(), t.imag.cpu(), 0)

        dtype_list = [torch.complex32, torch.complex64, torch.complex128]
        for dtype in dtype_list:
            fn(dtype)
            fn(dtype, contiguous_input=False)

            # tensor with zero elements
            x = torch.tensor([], dtype=dtype, device=torch.device("mlu:0"))
            res = torch.view_as_real(x)
            self.assertEqual(res.shape, torch.Size([0, 2]))

            # tensor with zero dim
            x = torch.tensor(2 + 3j, dtype=dtype, device=torch.device("mlu:0"))
            res = torch.view_as_real(x)
            self.assertEqual(res.shape, torch.Size([2]))


if __name__ == "__main__":
    unittest.main()
