import sys
import os
import unittest
import logging

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411


logging.basicConfig(level=logging.DEBUG)


class TestItem(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_item(self):
        type_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.long,
            torch.int8,
            torch.uint8,
            torch.bool,
            torch.cdouble,
            torch.cfloat,
            torch.chalf,
        ]
        for dtype in type_list:
            x = torch.ones((), dtype=dtype)
            out_cpu = x.item()
            out_mlu = x.mlu().item()
            self.assertEqual(out_cpu, out_mlu)
            self.assertTrue(type(out_cpu) == type(out_mlu))

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_item_bfloat16(self):
        type_list = [torch.bfloat16]
        for dtype in type_list:
            x = torch.ones((), dtype=dtype)
            out_cpu = x.item()
            out_mlu = x.mlu().item()
            self.assertEqual(out_cpu, out_mlu)
            self.assertTrue(type(out_cpu) == type(out_mlu))

    # @unittest.skip("not test")
    @testinfo()
    def test_item_exception(self):
        x = torch.randn([1, 2], dtype=torch.float)
        ref_msg = "a Tensor with 2 elements cannot be converted to Scalar"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x.mlu().item()


if __name__ == "__main__":
    unittest.main()
