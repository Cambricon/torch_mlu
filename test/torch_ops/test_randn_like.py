from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
from scipy import stats

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

# DEVICE_TYPE = ct.is_using_floating_device()
DEVICE_TYPE = True

logging.basicConfig(level=logging.DEBUG)


class TestRandn_like_Ops(TestCase):
    # @unittest.skip("not test")
    @unittest.skipIf(not DEVICE_TYPE, "only train on 300 MLU")
    @testinfo()
    def test_randn(self):
        SIZE = 100
        device = "mlu"
        dtypes = [torch.double, torch.float32, torch.half]
        for size in [0, SIZE]:
            for dtype in dtypes:
                res1 = torch.randn(size, size, dtype=dtype, device=device)
                torch.manual_seed(123456)
                res2 = torch.randn_like(res1)
                torch.manual_seed(123456)
                res3 = torch.randn_like(res1)
                self.assertEqual(res2, res3)

    # @unittest.skip("not test")
    @testinfo()
    def test_randn_like_device(self):
        device_indexes = range(torch.mlu.device_count())
        for device_index in device_indexes:
            device = "mlu:" + str(device_index)
            tensor = torch.randn(2, 3, device=device)
            device_tensor = torch.randn_like(tensor)
            self.assertEqual(device_tensor.device.index, device_index)

    # TODO(guwei): close the test case.
    @unittest.skip("not test")
    @testinfo()
    def test_randn_like_kstest(self):
        sizes = [15, 20, 50, 100, 150, 200, 250, 300, 350, 400]
        dtypes = [torch.double, torch.float32, torch.half]
        failure_count = 0
        total_count = 0
        for size in sizes:
            for dtype in dtypes:
                total_count += 1
                tensor = torch.randn(size, size, dtype=dtype, device="mlu")
                mlu_tensor = torch.randn_like(tensor)
                res = stats.kstest(mlu_tensor.cpu().numpy().reshape(-1), "norm")
                if res.statistic > 0.1:
                    failure_count += 1
        self.assertTrue(failure_count < total_count * 0.1)


if __name__ == "__main__":
    unittest.main()
