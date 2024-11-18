from __future__ import print_function

import sys
import json
import os
import io
import unittest
import logging

import torch
import torch_mlu
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.testing._internal.common_utils import TemporaryFileName


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)

os.environ["KINETO_MLU_USE_CONFIG"] = cur_dir + "/test_config.json"


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
    # @unittest.skip("not test")
    @testinfo()
    def test_profiler_blacklist(self):
        x = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ]
        ) as p:
            output = model(x)

        has_conv = False
        has_copy = False
        has_invokekernel = False
        has_sync = False

        for info in p.events():
            if "cnnlConvolutionForward" == info.name:
                has_conv = True
            if "cnnlCopy_v2" == info.name:
                has_copy = True
            if "cnInvokeKernel" == info.name:
                has_invokekernel = True
            if "cnrtSyncDevice" == info.name:
                has_sync = True
        self.assertFalse(has_conv)
        self.assertFalse(has_copy)
        self.assertFalse(has_invokekernel)
        self.assertFalse(has_sync)

    # @unittest.skip("not test")
    @testinfo()
    def test_profiler_whitelist(self):
        x = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ]
        ) as p:
            output = model(x)

        has_get_workspace = False
        has_descriptor = False
        has_set_queue = False

        for info in p.events():
            if "cnnlGetCopyWorkspaceSize" == info.name:
                has_get_workspace = True
            if "cnnlCreateTensorDescriptor" == info.name:
                has_descriptor = True
            if "cnnlSetQueue" == info.name:
                has_set_queue = True
        self.assertTrue(has_get_workspace)
        self.assertTrue(has_descriptor)
        self.assertTrue(has_set_queue)

    @staticmethod
    def _run_profiler_with_custom_env(rank, file_path, record_all, event_name_list):
        os.environ["KINETO_MLU_USE_CONFIG"] = file_path
        os.environ["KINETO_MLU_RECORD_ALL_APIS"] = record_all
        model = ConvNet()
        model = model.mlu()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ]
        ) as p:
            x = torch.randn(1, 3, 128, 128).mlu()
            _ = model(x)
        p.export_chrome_trace("tmp_trace.json")

        for event in p.events():
            event_name_list.append(event.name)

    @testinfo()
    def test_profiler_config_record_all(self):
        config_context = """
{
  "cnnl_whitelist": ["all", "cnnlSetQueue"],
  "cnrt_whitelist": ["", "all"],
  "cndrv_whitelist": ["all"]
}
"""
        with TemporaryFileName(mode="w+") as fname:
            with io.open(fname, "w") as f:
                f.write(config_context)

            # Use a new process to reset env
            with mp.Manager() as manager:
                # test all whitelist using config
                event_name_list = manager.list()
                mp.spawn(
                    TestProfiler._run_profiler_with_custom_env,
                    args=(fname, "0", event_name_list),
                    nprocs=1,
                    join=True
                )

                # test record all using KINETO_MLU_RECORD_ALL_APIS and set config to empty
                event_name_list2 = manager.list()
                mp.spawn(
                    TestProfiler._run_profiler_with_custom_env,
                    args=("", "1", event_name_list2),
                    nprocs=1,
                    join=True
                )
                self.assertEqual(list(event_name_list).sort(), list(event_name_list2).sort())

                found_cnnl_descritor = False
                found_cnnl_workspace_size = False
                found_cnnl_conv = False
                # Check apis in black list by default
                some_cnrt_apis = ["cnrtGetDevice", "cnrtGetLastError", "cnrtInvokeKernel", "cnrtSetDevice", "cnrtQueueSync"]
                some_cndrv_apis = ["cnSharedContextAcquire", "cnCtxGetCurrent", "cnCtxSync", "cnCtxSetCurrent"]
                for name in event_name_list:
                    if name.startswith("cnnl") and "Descriptor" in name:
                        found_cnnl_descritor = True
                    if name.startswith("cnnl") and "WorkspaceSize" in name:
                        found_cnnl_workspace_size  = True
                    if name.startswith("cnnl") and "Convolution" in name:
                        found_cnnl_conv = True
                self.assertTrue(found_cnnl_descritor)
                self.assertTrue(found_cnnl_workspace_size)
                self.assertTrue(found_cnnl_conv)
                event_name_set = set(event_name_list)
                self.assertTrue(set(some_cnrt_apis).issubset(event_name_set))
                self.assertTrue(set(some_cndrv_apis).issubset(event_name_set))

    @testinfo()
    def test_profiler_config_only_record_cndrv_all(self):
        config_context = """
{
  "cnnl_blacklist": ["all", "cnnlSetQueue"],
  "cnrt_blacklist": ["", "all"],
  "cndrv_whitelist": ["all"]
}
"""
        with TemporaryFileName(mode="w+") as fname:
            with io.open(fname, "w") as f:
                f.write(config_context)

            # Use a new process to reset env
            with mp.Manager() as manager:
                event_name_list = manager.list()
                mp.spawn(
                    TestProfiler._run_profiler_with_custom_env,
                    args=(fname, "0", event_name_list),
                    nprocs=1,
                    join=True
                )

                found_cnnl = False
                found_cnrt = False
                some_cndrv_apis = ["cnSharedContextAcquire", "cnCtxGetCurrent", "cnCtxSync", "cnCtxSetCurrent"]
                for name in event_name_list:
                    if name.startswith("cnnl"):
                        found_cnnl = True
                    if name.startswith("cnrt"):
                        found_cnrt  = True
                # Should not find cnnl and cnrt
                self.assertFalse(found_cnnl)
                self.assertFalse(found_cnrt)
                event_name_set = set(event_name_list)
                self.assertTrue(set(some_cndrv_apis).issubset(event_name_set))

    @testinfo()
    def test_profiler_cnnl_exclude_whitelist(self):
        config_context = """
{
  "cnnl_whitelist": ["cnnlCreate", "cnnlSetConvolutionDescriptor", "cnnlGetPoolingWithIndexWorkspaceSize_v2"]
}
"""
        with TemporaryFileName(mode="w+") as fname:
            with io.open(fname, "w") as f:
                f.write(config_context)

            # Use a new process to reset env
            with mp.Manager() as manager:
                event_name_list = manager.list()
                mp.spawn(
                    TestProfiler._run_profiler_with_custom_env,
                    args=(fname, "0", event_name_list),
                    nprocs=1,
                    join=True
                )

                should_found_cnnl_apis = ["cnnlCreate", "cnnlSetConvolutionDescriptor", "cnnlGetPoolingWithIndexWorkspaceSize_v2",
                                          "cnnlPoolingForwardWithIndex", "cnnlConvolutionForward"]
                should_not_found_cnnl_apis = ["cnnlSetQueue",
                                              "cnnlCreateConvolutionDescriptor",
                                              "cnnlGetConvolutionForwardWorkspaceSize"]
                event_name_set = set(event_name_list)
                self.assertTrue(set(should_found_cnnl_apis).issubset(event_name_set))
                self.assertTrue(event_name_set.isdisjoint(set(should_not_found_cnnl_apis)))

if __name__ == "__main__":
    unittest.main()
