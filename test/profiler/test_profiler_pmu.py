from __future__ import print_function

import sys
import json
import os
import unittest
from unittest.mock import patch
from itertools import product
import logging

import torch
import torch_mlu
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.testing._internal.common_utils import TemporaryFileName

from torch.profiler import profile

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x


class TestProfiler(TestCase):
    @staticmethod
    def _run_profiler(rank, enable_with_env, enable_with_config):
        kwargs = {}
        if enable_with_config:
            kwargs["experimental_config"] = torch._C._profiler._ExperimentalConfig(
                profiler_metrics=["tp_core__read_bytes", "llc__write_bytes"],
                profiler_measure_per_kernel=True,
            )
        if enable_with_env:
            os.environ["ENABLE_CATCHING_PMU_DATA"] = "1"
        model = ConvNet()
        model = model.mlu()
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ],
            **kwargs
        ) as p:
            x = torch.randn(1, 3, 128, 128).mlu()
            model(x)
        pmu_names = []
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]
            kernels = [e for e in events if "cat" in e.keys() and e["cat"] == "kernel"]
            first_kernel = True
            for kernel in kernels:
                args = kernel["args"]
                if "pmus" in args.keys():
                    if first_kernel:
                        pmu_names = args["pmus"].keys()
                        first_kernel = False
                    else:
                        assert set(pmu_names) == set(
                            args["pmus"].keys()
                        ), "All kernels should contain same pmu names"

        if enable_with_config and enable_with_env:
            # Using CONFIG has higher priority
            assert set(pmu_names) == set(["tp_core__read_bytes", "llc__write_bytes"])
        elif enable_with_config:
            assert set(pmu_names) == set(["tp_core__read_bytes", "llc__write_bytes"])
        elif enable_with_env:
            # Assert some default value:tp_core__lt_cycles,tp_cluster__write_bytes
            assert len(pmu_names) > 2
            assert "tp_core__lt_cycles" in pmu_names
            assert "tp_cluster__write_bytes" in pmu_names
            assert "llc__write_bytes" not in pmu_names
        else:
            assert len(pmu_names) == 0

    @staticmethod
    def _run_multi_profiler(rank, pmu_cached):
        enabled_counters_1 = ["tp_core__read_bytes", "llc__write_bytes"]
        enabled_counters_2 = ["tp_core__write_bytes", "llc__read_bytes"]
        model = ConvNet()
        model = model.mlu()
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ],
            experimental_config=torch._C._profiler._ExperimentalConfig(
                profiler_metrics=enabled_counters_1,
                profiler_measure_per_kernel=True,
            ),
        ) as p1:
            x = torch.randn(1, 3, 128, 128).mlu()
            model(x)

        # Run another profiler with different config in the same thread
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ],
            experimental_config=torch._C._profiler._ExperimentalConfig(
                profiler_metrics=enabled_counters_2,
                profiler_measure_per_kernel=True,
            ),
        ) as p2:
            x = torch.randn(1, 3, 128, 128).mlu()
            model(x)

        def _get_pmu_names(p):
            pmu_names = []
            with TemporaryFileName(mode="w+") as fname:
                p.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                kernels = [
                    e for e in events if "cat" in e.keys() and e["cat"] == "kernel"
                ]
                first_kernel = True
                for kernel in kernels:
                    args = kernel["args"]
                    if "pmus" in args.keys():
                        if first_kernel:
                            pmu_names = args["pmus"].keys()
                            first_kernel = False
                        else:
                            assert set(pmu_names) == set(
                                args["pmus"].keys()
                            ), "All kernels should contain same pmu names"
            return pmu_names

        pmu_names_1 = _get_pmu_names(p1)
        pmu_names_2 = _get_pmu_names(p2)
        if pmu_names_1:
            assert set(pmu_names_1) == set(enabled_counters_1)
            pmu_cached.append(True)
        else:
            pmu_cached.append(False)
        if pmu_names_2:
            assert set(pmu_names_2) == set(enabled_counters_2)
            pmu_cached.append(True)
        else:
            pmu_cached.append(False)

    @testinfo()
    def test_profiler_generate_pmu_data(self):
        for enable_env, enable_config in product([True, False], [True, False]):
            # Use a new process to reset env and config.
            mp.spawn(
                TestProfiler._run_profiler,
                args=(enable_env, enable_config),
                nprocs=1,
                join=True,
            )

    @testinfo()
    def test_enable_pmu_with_different_config(self):
        with mp.Manager() as manager:
            pmu_cached = manager.list()
            mp.spawn(
                TestProfiler._run_multi_profiler,
                args=(pmu_cached,),
                nprocs=1,
                join=True,
            )
            # Run 2 profilers in _run_multi_profiler() so the count of 'True' is 2.
            self.assertEqual(pmu_cached.count(True), 2)
            self.assertEqual(pmu_cached.count(False), 0)

    @testinfo()
    @unittest.skip("SYSTOOL-4993")
    def test_multi_process_enable_pmu(self):
        with mp.Manager() as manager:
            pmu_cached = manager.list()
            mp.spawn(
                TestProfiler._run_multi_profiler,
                args=(pmu_cached,),
                nprocs=4,
                join=True,
            )
            self.assertGreaterEqual(pmu_cached.count(True), 2)


if __name__ == "__main__":
    unittest.main()
