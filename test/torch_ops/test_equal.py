from __future__ import print_function

import sys
import os
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0411, C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestEqualOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_equal(self):
        for shape1, shape2 in [
            ((1, 144, 7, 15, 2, 1, 1, 2), (256, 144, 7, 15, 2, 1, 1, 1)),
            ((1, 144, 1, 1, 1, 1, 1, 1), (256, 1, 7, 15, 2, 1, 1, 2)),
            ((5), (5)),
            ((4), (5)),
            ((), ()),
            ((2, 3, 4), (3, 4)),
            ((1, 117, 1, 4, 1, 5, 1, 2), (117, 1, 5, 1, 5, 1, 3, 1)),
            ((256, 144, 7, 15, 2, 1, 1, 1), (1)),
        ]:
            x = torch.randn(shape1, dtype=torch.float)
            y = torch.randn(shape2, dtype=torch.float)
            out_cpu = torch.equal(x, y)
            out_mlu = torch.equal(self.to_mlu(x), self.to_mlu(y))
            self.assertEqual(out_cpu, out_mlu, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_equal_channels_last(self):
        for shape1, shape2 in [
            ((2, 3, 24, 30), (1, 1, 1, 30)),
            ((16, 8, 8, 32), (16, 8, 8, 32)),
        ]:
            x = torch.randn(shape1, dtype=torch.float).to(
                memory_format=torch.channels_last
            )
            y = torch.randn(shape2, dtype=torch.float).to(
                memory_format=torch.channels_last
            )
            out_cpu = torch.equal(x, y)
            out_mlu = torch.equal(self.to_mlu(x), self.to_mlu(y))
            self.assertEqual(out_cpu, out_mlu, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_equal_not_dense(self):
        for shape1, shape2 in [
            ((2, 3, 24, 30), (1, 1, 1, 30)),
            ((16, 8, 8, 32), (16, 8, 8, 32)),
        ]:
            x = torch.randn(shape1, dtype=torch.float)[:, :, :, :15]
            y = torch.randn(shape2, dtype=torch.float)[:, :, :, :15]
            out_cpu = torch.equal(x, y)
            out_mlu = torch.equal(self.to_mlu(x), self.to_mlu(y))
            self.assertEqual(out_cpu, out_mlu, 0)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_equal_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 3, 4, 6), dtype=torch.bfloat16, device="cpu"
        )
        right = torch.testing.make_tensor((3, 4, 6), dtype=torch.bfloat16, device="cpu")
        out_cpu = torch.equal(left, right)
        out_mlu = torch.equal(left.mlu(), right.mlu())
        self.assertEqual(out_cpu, out_mlu, 0)


if __name__ == "__main__":
    unittest.main()
