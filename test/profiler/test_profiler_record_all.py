from __future__ import print_function

import sys
import json
import os
import unittest
import logging

import torch
import torch_mlu
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import (
    profile,
    record_function,
    supported_activities,
    DeviceType,
    ProfilerAction,
    ProfilerActivity,
)

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)

os.environ["KINETO_MLU_RECORD_ALL_APIS"] = "true"


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
        has_cnrt_invoke = False
        has_cnmodule_load = False

        for info in p.events():
            if "cnnlGetCopyWorkspaceSize" == info.name:
                has_get_workspace = True
            if "cnnlCreateTensorDescriptor" == info.name:
                has_descriptor = True
            if "cnnlSetQueue" == info.name:
                has_set_queue = True
            if "cnrtInvokeKernel" == info.name:
                has_cnrt_invoke = True
            if "cnModuleLoadFatBinary" == info.name:
                has_cnmodule_load = True
        self.assertTrue(has_get_workspace)
        self.assertTrue(has_descriptor)
        self.assertTrue(has_set_queue)
        self.assertTrue(has_cnrt_invoke)
        self.assertTrue(has_cnmodule_load)


if __name__ == "__main__":
    unittest.main()
