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


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_reshape(self):
        for in_shape, out_shape in [
            ((1, 1000, 1, 1), (1, -1)),
            ((1, 3, 200, 200), (1, -1, 1, 200, 200)),
            ((1,), (1,)),
            ((1, 58, 2, 28, 28), (2, -1, 4)),
            ((45, 54, 454), (45, 54, 454)),
        ]:
            x_cpu = torch.randint(0, 100, (in_shape))
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu.reshape(out_shape)
            y_mlu = x_mlu.reshape(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_reshape_channels_last(self):
        for in_shape, out_shape in [
            ((1, 1000, 1, 1), (1, -1)),
            ((2, 3, 4, 5), (2, -1, 1, 4, 5)),
        ]:
            x_cpu = torch.randn(in_shape).to(memory_format=torch.channels_last)
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu.reshape(out_shape)
            y_mlu = x_mlu.reshape(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_reshape_channels_last_not_dense(self):
        for in_shape, out_shape in [
            ((1, 1000, 4, 3), (1, -1)),
            ((2, 5, 4, 5), (2, -1, 1, 4, 5)),
        ]:
            x_cpu = torch.randn(in_shape).to(memory_format=torch.channels_last)
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu[:, :2].reshape(out_shape)
            y_mlu = x_mlu[:, :2].reshape(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_reshape_channels_last_by_copy(self):
        for in_shape, out_shape in [
            ((64, 3, 24, 24), (1, -1)),
            ((2, 3, 4, 5), (-1, 5)),
            ((13, 77, 23, 153), (23, 1001, 153)),
        ]:
            x_cpu = torch.randn(in_shape)
            x_cl = x_cpu.to(memory_format=torch.channels_last)
            x_mlu = self.to_device(x_cl)
            y_cpu = x_cpu.reshape(out_shape)
            y_mlu = x_mlu.reshape(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cl.size() == x_mlu.size())
            self.assertTrue(x_cl.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_reshape_bfloat16(self):
        for in_shape, out_shape in [
            ((1, 1000, 1, 1), (1, -1)),
            ((1, 3, 200, 200), (1, -1, 1, 200, 200)),
            ((1,), (1,)),
            ((1, 58, 2, 28, 28), (2, -1, 4)),
            ((45, 54, 454), (45, 54, 454)),
        ]:
            x_cpu = torch.randint(0, 100, (in_shape), dtype=torch.bfloat16)
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu.reshape(out_shape)
            y_mlu = x_mlu.reshape(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)


if __name__ == "__main__":
    unittest.main()
