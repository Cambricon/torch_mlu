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


class TestDataLoader(TestCase):
    def setUp(self):
        super(TestDataLoader, self).setUp()
        self.data = torch.randn(100, 3, 10, 10)
        self.labels = torch.randn(100)
        self.dataset = TensorDataset(self.data, self.labels)

    def _test_generator(
        self, num_workers, batch_size, pin_memory, pin_memory_device, non_blocking
    ):
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=pin_memory_device,
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
            for pin_memory_device in ["", "mlu"]:
                for num_workers in [0, 2]:  # 0 is single process; 1 is multi process
                    for non_blocking in [True, False]:
                        for batch_size in [1, 2]:
                            self._test_generator(
                                num_workers,
                                batch_size,
                                pin_memory,
                                pin_memory_device,
                                non_blocking,
                            )


if __name__ == "__main__":
    unittest.main()
