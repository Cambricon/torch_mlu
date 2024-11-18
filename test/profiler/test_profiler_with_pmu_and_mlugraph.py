from __future__ import print_function

import sys
import json
import os
import unittest
import logging

import torch
import torch_mlu

from torch.testing._internal.common_utils import TemporaryFileName

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)

os.environ["ENABLE_CATCHING_PMU_DATA"] = "1"
os.environ["TORCH_MLU_ENABLE_CATCHING_MLUGRAPH_OP"] = "1"


class TestProfiler(TestCase):
    @testinfo()
    def test_profiler_capture_and_pmu_enabled(self):
        g = torch.mlu.MLUGraph()
        x = torch.randn(1, 64, 32, 256, device="mlu")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ],
        ) as p:
            a = torch.randn((64, 128)).mlu()
            b = torch.randn((64, 128)).mlu()
            c = a + b

            with torch.mlu.graph(g):
                y = x.clone()
                for i in range(5):
                    y = torch.nn.functional.gelu(y)
            xm = torch.randn(1, 64, 32, 256, device="mlu")
            x.copy_(xm)

            g.replay()

        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]

            # c = a + b will launch optensor kernel, this should have pmus
            found_optensor_with_pmu = False
            # gelu op in MLUGraph that shouldn't have pmus
            found_gule_without_pmu = False
            kernels = [e for e in events if "cat" in e.keys() and e["cat"] == "kernel"]
            for kernel in kernels:
                args = kernel["args"]
                if "pmus" in args.keys() and "OpTensor" in kernel["name"]:
                    found_optensor_with_pmu = True
                if (
                    "tasktopo external op" in args.keys()
                    and "aten::gelu" in args["tasktopo external op"]
                    and "pmus" not in args.keys()
                ):
                    found_gule_without_pmu = True

            self.assertTrue(found_optensor_with_pmu)
            self.assertTrue(found_gule_without_pmu)


if __name__ == "__main__":
    unittest.main()
