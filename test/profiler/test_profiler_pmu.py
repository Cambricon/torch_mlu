from __future__ import print_function

import sys
import json
import os
import unittest
from unittest.mock import patch
import logging

import torch
import torch_mlu

from torch.testing._internal.common_utils import TemporaryFileName

from torch.profiler import profile

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)


class TestProfiler(TestCase):
    def payload(self, use_mlu=False):
        x = torch.randn(10, 10)
        if use_mlu:
            x = x.to("mlu")
        y = torch.randn(10, 10)
        if use_mlu:
            y = y.to("mlu")
        z = torch.mm(x, y)
        z = z + y
        if use_mlu:
            z = z.cpu()

    @testinfo()
    @patch.dict(os.environ, {"ENABLE_CATCHING_PMU_DATA": "1"})
    def test_profiler_generate_pmu_data(self):
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ]
        ) as p:
            self.payload(use_mlu=True)

        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]

            pmu_counter_nums = []
            kernels = [e for e in events if "cat" in e.keys() and e["cat"] == "kernel"]
            for kernel in kernels:
                args = kernel["args"]
                self.assertTrue("pmus" in args.keys())
                pmus = args["pmus"]
                pmu_counter_nums.append(len(pmus))

            self.assertTrue(len(pmu_counter_nums) > 0)
            self.assertTrue(pmu_counter_nums[0] > 0)
            self.assertTrue(
                all([num == pmu_counter_nums[0] for num in pmu_counter_nums])
            )


if __name__ == "__main__":
    unittest.main()
