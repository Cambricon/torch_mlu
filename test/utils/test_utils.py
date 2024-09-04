from __future__ import print_function
import torch
import torch.nn as nn
import torch_mlu
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import sys
import os
import copy

import time
import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
import logging

logging.basicConfig(level=logging.DEBUG)


class TestQueue(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_cmake_prefix_path(self):
        import torch.utils as utils
        import torch_mlu.utils as mlu_utils

        torch_path = utils.cmake_prefix_path
        torch_mlu_path = mlu_utils.cmake_prefix_path
        print("TORCH_PATH: ")
        print(torch_path)
        print("TORCH_MLU_PATH: ")
        print(torch_mlu_path)

    # @unittest.skip("not test")
    @testinfo()
    def test_shell_cmake_prefix_path(self):
        import os

        torch_cmd = (
            "python -c 'import torch.utils as utils;print(utils.cmake_prefix_path)'"
        )
        torch_mlu_cmd = "python -c 'import torch_mlu.utils as mlu_utils;print(mlu_utils.cmake_prefix_path)'"
        print("TORCH_PATH: ")
        os.system(torch_cmd)
        print("TORCH_MLU_PATH: ")
        os.system(torch_mlu_cmd)


class TestVersion(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_get_version(self):
        print("TORCH MLU VERSION:")
        print(torch_mlu.get_version())
        print(torch_mlu.__version__)
        print(torch_mlu.get_git_version())


if __name__ == "__main__":
    unittest.main()
