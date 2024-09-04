from __future__ import print_function
import sys
import os
from sys import path
from os.path import dirname
import unittest
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


class TestPinMemory(TestCase):
    def setUp(self):
        super(TestPinMemory, self).setUp()
        self.data = torch.randn(100, 3, 10, 10)
        # TODO:copy with Long dtype and non_blocking throw exception,
        # because not support empty_stride current.
        # self.labels = torch.randperm(50).repeat(2)
        self.labels = torch.randn(100)
        self.dataset = TensorDataset(self.data, self.labels)

    def _test_generator(self, num_workers, pin_memory, non_blocking):
        loader = DataLoader(
            self.dataset, batch_size=2, num_workers=num_workers, pin_memory=pin_memory
        )
        for input_t, target in loader:
            self.assertTrue(pin_memory == input_t.is_pinned())
            self.assertTrue(pin_memory == target.is_pinned())

            input_mlu = input_t.to("mlu", non_blocking=non_blocking)
            target_mlu = target.to("mlu", non_blocking=non_blocking)
            input_mlu *= 1
            target_mlu *= 1
            input_cpu = input_mlu.cpu()
            target_cpu = target_mlu.cpu()
            self.assertTensorsEqual(input_t.cpu(), input_cpu, 0.0, use_MSE=True)
            self.assertTensorsEqual(target.cpu(), target_cpu, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_dataloader(self):
        for pin_memory in [True, False]:
            for num_workers in [0, 1]:  # 0 is single process; 1 is multi process
                for non_blocking in [True, False]:
                    self._test_generator(num_workers, pin_memory, non_blocking)

    # @unittest.skip("not test")
    @testinfo()
    def test_pin_memory(self):
        for shape in [(1, 3, 224, 224), (2, 30, 80), (3, 20), (10), (1, 3, 224), (1)]:
            for pin_memory in [True, False]:
                for non_blocking in [True, False]:
                    x = torch.randn(shape)
                    if pin_memory:
                        x = x.pin_memory()
                    self.assertTrue(pin_memory == x.is_pinned())
                    x_mlu = x.to("mlu", non_blocking=non_blocking)
                    self.assertTensorsEqual(x_mlu.cpu(), x, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pin_memory_device(self):
        for shape in [(1, 3, 224, 224), (2, 30, 80), (3, 20), (10), (1, 3, 224), (1)]:
            for pin_memory in [True, False]:
                for non_blocking in [True, False]:
                    x = torch.randn(shape)
                    if pin_memory:
                        x = x.pin_memory(device="mlu")
                    self.assertTrue(pin_memory == x.is_pinned(device="mlu"))
                    x_mlu = x.to("mlu", non_blocking=non_blocking)
                    self.assertTensorsEqual(x_mlu.cpu(), x, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test__pin_memory(self):
        for shape in [(1, 3, 224, 224), (2, 30, 80), (3, 20), (10), (1, 3, 224), (1)]:
            x = torch.randn(shape)
            out = torch._pin_memory(x)
            self.assertTrue(out.is_pinned())
            out_mlu = out.to("mlu", non_blocking=True)
            self.assertTensorsEqual(out_mlu.cpu(), x, 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_pin_memory_slice(self):
        x = torch.randn(16, 3, 5, 7)
        x = x.pin_memory()
        self.assertTrue(x.is_pinned())
        x_mlu = x.to("mlu", non_blocking=True)
        self.assertTensorsEqual(x_mlu.cpu(), x, 0.0, use_MSE=True)
        y = x[:8, :, :, :]
        self.assertTrue(y.is_pinned())

    # @unittest.skip("not test")
    @testinfo()
    def test_pin_memory_pinned(self):
        x = torch.randn(3, 5)
        self.assertFalse(x.is_pinned())
        if not torch.mlu.is_available():
            self.assertRaises(RuntimeError, x.pin_memory())
        else:
            pinned = x.pin_memory()
            self.assertTrue(pinned.is_pinned())
            self.assertEqual(pinned, x)
            self.assertNotEqual(pinned.data_ptr(), x.data_ptr())
            self.assertIs(pinned, pinned.pin_memory())
            self.assertEqual(pinned.data_ptr(), pinned.pin_memory().data_ptr())


if __name__ == "__main__":
    unittest.main()
