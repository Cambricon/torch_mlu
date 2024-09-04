import logging
import os
import sys
import unittest
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestDistributions(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_manual_seed(self):
        # When not set maunal seed, the generated random numbers is different,
        # after seting the seed ,the random numbers should be stable.
        a = torch.randn(1, dtype=torch.float).to("mlu")
        a.uniform_()
        b = torch.randn(1, dtype=torch.float).to("mlu")
        b.uniform_()
        self.assertNotEqual(a.cpu(), b.cpu())

        torch.manual_seed(1)
        c = torch.randn(10, dtype=torch.float).to("mlu")
        c.uniform_()
        torch.manual_seed(1)
        d = torch.randn(10, dtype=torch.float).to("mlu")
        d.uniform_()
        self.assertEqual(c.cpu(), d.cpu())

        torch.manual_seed(42)
        c = torch.randn(10, dtype=torch.float).to("mlu")
        c.bernoulli_()
        torch.manual_seed(42)
        d = torch.randn(10, dtype=torch.float).to("mlu")
        d.bernoulli_()
        self.assertEqual(c.cpu(), d.cpu())


if __name__ == "__main__":
    unittest.main()
