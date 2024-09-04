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
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestRandpermOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_randperm(self):
        for dtype in [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.half,
            torch.int32,
            torch.float32,
            torch.long,
            torch.double,
        ]:
            for n in [1, 10, 100]:
                out_mlu = torch.randperm(n, dtype=dtype, device="mlu")
                out_cpu = torch.randperm(n, dtype=dtype, device="cpu")
                self.assertEqual(out_mlu.numel(), n)
                self.assertEqual(out_mlu.dtype, dtype)
                self.assertEqual(out_mlu.device, torch.device("mlu", index=0))
                self.assertLess(
                    abs(
                        out_mlu.cpu().float().sum().item()
                        - out_cpu.float().sum().item()
                    ),
                    0.003,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_randperm_out(self):
        for n in [1, 100]:
            out_cpu = torch.randn(n + 1)
            out_mlu = torch.randn(n + 1).mlu()
            origin_ptr = out_mlu.data_ptr()
            torch.randperm(n, out=out_mlu, device="mlu")
            torch.randperm(n, out=out_cpu, device="cpu")
            self.assertEqual(out_mlu.numel(), n)
            self.assertEqual(out_mlu.device, torch.device("mlu", index=0))
            self.assertEqual(origin_ptr, out_mlu.data_ptr())
            self.assertLess(
                abs(out_mlu.cpu().float().sum().item() - out_cpu.float().sum().item()),
                0.003,
            )
            out_cpu = torch.randn(n - 1)
            out_mlu = torch.randn(n - 1).mlu()
            torch.randperm(n, out=out_mlu, device="mlu")
            torch.randperm(n, out=out_cpu, device="cpu")
            self.assertEqual(out_mlu.numel(), n)
            self.assertEqual(out_mlu.device, torch.device("mlu", index=0))
            self.assertLess(
                abs(out_mlu.cpu().float().sum().item() - out_cpu.float().sum().item()),
                0.003,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_randperm_generator(self):
        for dtype in [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.half,
            torch.int32,
            torch.float32,
            torch.long,
            torch.double,
        ]:
            g = torch.Generator(device="mlu")
            out_mlu = torch.randperm(10, generator=g, dtype=dtype, device="mlu")
            out_cpu = torch.randperm(10, generator=g, dtype=dtype, device="cpu")
            self.assertEqual(out_mlu.numel(), 10)
            self.assertEqual(out_mlu.dtype, dtype)
            self.assertEqual(out_mlu.device, torch.device("mlu", index=0))
            self.assertLess(
                abs(out_mlu.cpu().float().sum().item() - out_cpu.float().sum().item()),
                0.003,
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_randperm_bfloat16(self):
        dtype = torch.bfloat16
        for n in [1, 10, 100]:
            out_mlu = torch.randperm(n, dtype=dtype, device="mlu")
            out_cpu = torch.randperm(n, dtype=dtype, device="cpu")
            self.assertEqual(out_mlu.numel(), n)
            self.assertEqual(out_mlu.dtype, dtype)
            self.assertEqual(out_mlu.device, torch.device("mlu", index=0))
            self.assertLess(
                abs(out_mlu.cpu().float().sum().item() - out_cpu.float().sum().item()),
                0.003,
            )


if __name__ == "__main__":
    unittest.main()
