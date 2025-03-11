from __future__ import print_function

import logging
import unittest
import sys
import os
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase

logging.basicConfig(level=logging.DEBUG)


class TestOpPrecisionMode(TestCase):
    @testinfo()
    def test_mlu_precision_mode_get_set(self):
        op_list = torch.mlu.get_precision_supported_op_list()
        self.assertTrue(len(op_list) > 0)

        op = op_list[0]
        orig_mode = torch.mlu.get_precision_mode(op)
        with torch.mlu.precision_mode("low", op):
            self.assertEqual(torch.mlu.get_precision_mode(op), "low")
        self.assertEqual(torch.mlu.get_precision_mode(op), orig_mode)
        with torch.mlu.precision_mode("high", op):
            self.assertEqual(torch.mlu.get_precision_mode(op), "high")
        self.assertEqual(torch.mlu.get_precision_mode(op), orig_mode)

        torch.mlu.set_precision_mode("low", op)
        self.assertEqual(torch.mlu.get_precision_mode(op), "low")
        torch.mlu.set_precision_mode("high", op)
        self.assertEqual(torch.mlu.get_precision_mode(op), "high")

    @testinfo()
    def test_mlu_precision_mode_for_all_op(self):
        op_list = torch.mlu.get_precision_supported_op_list()
        self.assertTrue(len(op_list) > 0)

        orig_modes = {}
        for op in op_list:
            orig_modes[op] = torch.mlu.get_precision_mode(op)
        with torch.mlu.precision_mode("low", "all_op"):
            for op in op_list:
                self.assertEqual(torch.mlu.get_precision_mode(op), "low")
        for op in op_list:
            self.assertEqual(torch.mlu.get_precision_mode(op), orig_modes[op])
        with torch.mlu.precision_mode("high", "all_op"):
            for op in op_list:
                self.assertEqual(torch.mlu.get_precision_mode(op), "high")
        for op in op_list:
            self.assertEqual(torch.mlu.get_precision_mode(op), orig_modes[op])

        torch.mlu.set_precision_mode("low")
        for op in op_list:
            self.assertEqual(torch.mlu.get_precision_mode(op), "low")
        torch.mlu.set_precision_mode("high", "all_op")
        for op in op_list:
            self.assertEqual(torch.mlu.get_precision_mode(op), "high")

    @testinfo()
    def test_mlu_precision_mode_environ_var(self):
        # must set environ var first!
        os.environ["TORCH_OP_PRECISION_CONFIG"] = "silu: low, all_op: high"
        op_list = torch.mlu.get_precision_supported_op_list()
        self.assertTrue(len(op_list) > 0)
        for op in op_list:
            if op == "silu":
                self.assertEqual(torch.mlu.get_precision_mode(op), "low")
            else:
                self.assertEqual(torch.mlu.get_precision_mode(op), "high")

        # modifying environ var takes no effect
        os.environ["TORCH_OP_PRECISION_CONFIG"] = "silu: high"
        self.assertEqual(torch.mlu.get_precision_mode("silu"), "low")


if __name__ == "__main__":
    unittest.main()
