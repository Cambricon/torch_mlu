from __future__ import print_function
import logging
import unittest
import sys
import os
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase, testinfo, read_card_info, skipBFloat16IfNotSupport

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestExponentialOp(TestCase):
    def _test_exponential(self, shape, lambd, dtype):
        x = torch.randn(shape, dtype=dtype, device="mlu")
        x.exponential_(lambd=lambd)
        self.assertGreater(x.min(), 0)

        y = torch.randn(shape, dtype=dtype, device="mlu")
        torch.manual_seed(123)
        x.exponential_(lambd=lambd)
        torch.manual_seed(123)
        y.exponential_(lambd=lambd)
        self.assertEqual(x, y)

    # @unittest.skip("not test")
    @testinfo()
    def test_exponential(self):
        for dtype in [torch.float, torch.half, torch.double]:
            for shape in [[], [1], [2, 3], [2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 5, 6]]:
                for lambd in [0.1, 1.0, 1.2, 8.8]:
                    self._test_exponential(shape, lambd, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_exponential_PYTORCH_11152(self):
        x = torch.randn((1, 4, 1, 64, 64), dtype=torch.float, device="mlu")
        x.as_strided_(x.size(), stride=(4, 1, 4, 256, 4))
        x.exponential_(lambd=0.1)
        self.assertGreater(x.min(), 0)

        y = torch.randn((1, 4, 1, 64, 64), dtype=torch.float, device="mlu")
        y.as_strided_(y.size(), stride=(16384, 1, 16384, 256, 4))
        torch.manual_seed(123)
        x.exponential_(lambd=0.1)
        torch.manual_seed(123)
        y.exponential_(lambd=0.1)
        self.assertEqual(x, y)

    # @unittest.skip("not test")
    @testinfo()
    def test_exponential_lambda(self):
        x = torch.randn((1), dtype=torch.float, device="mlu")
        x.exponential_(lambd=float("inf"))
        self.assertEqual(x.cpu(), torch.tensor([0.0]))

    # @unittest.skip("not test")
    @testinfo()
    def test_exponential_lambda_exception(self):
        ref_msg = f"expects lambda > 0.0, but found lambda"
        for lambd in [-1.2]:
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                self._test_exponential([2, 3, 4], lambd, torch.float)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_exponential_bfloat16(self):
        self._test_exponential([2, 3], 0.1, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
