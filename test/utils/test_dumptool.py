from __future__ import print_function

# pylint: disable=C0411, C0413, W0612
import sys
import logging
import os
import shutil
import torch
import unittest
from torch_mlu.utils.dumptool import dump_cnnl_gencase

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase

logging.basicConfig(level=logging.DEBUG)


class TestDumptool(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_dump_cnnl_gencase(self):
        a = torch.randn((2, 2, 6, 6))
        b = torch.randn((2, 2, 6, 6))

        gencase_files = "./gen_case"
        for level in ["L1", "L2"]:
            if os.path.exists(gencase_files):
                shutil.rmtree(gencase_files)
                print("Remove gencase files: ", gencase_files)
            dump_cnnl_gencase(enable=True, level=level)
            out = (a.to("mlu") + b.to("mlu")).view(4, 36)
            dump_cnnl_gencase(enable=False)
            self.assertTrue(os.path.exists(gencase_files))

        for level in ["L3"]:
            dump_cnnl_gencase(enable=True, level=level)
            out = (a.to("mlu") + b.to("mlu")).view(4, 36)
            dump_cnnl_gencase(enable=False)

    # @unittest.skip("not test")
    @testinfo()
    def test_dump_cnnl_gencase_invalid_use(self):
        a = torch.randn((2, 2))
        b = torch.randn((2, 2))
        for level in ["L0", "l1", "L4"]:
            with self.assertRaises(ValueError):
                dump_cnnl_gencase(enable=True, level=level)
                out = a.to("mlu") + b.to("mlu")
                dump_cnnl_gencase(enable=False)


if __name__ == "__main__":
    unittest.main()
