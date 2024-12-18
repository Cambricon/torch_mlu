import os
import sys
import logging
import unittest
import numpy as np
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestDynamicPartitionOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_simple_1d(self):
        partitions = torch.tensor([0, 0, 2, 3, 2, 1], dtype=torch.int32, device="mlu")
        data = torch.tensor([0, 13, 2, 39, 4, 17], dtype=torch.float32, device="mlu")
        num_partitions = 4
        device_result = torch.ops.torch_mlu.dynamic_partition(
            data, partitions, num_partitions
        )
        # expected out
        expected_out = [
            torch.tensor([0, 13], dtype=torch.float32, device="cpu"),
            torch.tensor([17], dtype=torch.float32, device="cpu"),
            torch.tensor([2, 4], dtype=torch.float32, device="cpu"),
            torch.tensor([39], dtype=torch.float32, device="cpu"),
        ]
        self.assertEqual(len(device_result), num_partitions)
        for i in range(num_partitions):
            self.assertEqual(expected_out[i], device_result[i].cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_simple_2d(self):
        dtype_list = [torch.float, torch.int, torch.long]
        for dtype in dtype_list:
            partitions = torch.tensor(
                [0, 0, 2, 3, 2, 1], dtype=torch.int32, device="mlu"
            )
            data = torch.tensor(
                [
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                    [9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17],
                ],
                dtype=dtype,
                device="mlu",
            )
            num_partitions = 4
            device_result = torch.ops.torch_mlu.dynamic_partition(
                data, partitions, num_partitions
            )
            # expected out
            expected_out = [
                torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=dtype, device="cpu"),
                torch.tensor([[15, 16, 17]], dtype=dtype, device="cpu"),
                torch.tensor([[6, 7, 8], [12, 13, 14]], dtype=dtype, device="cpu"),
                torch.tensor([[9, 10, 11]], dtype=dtype, device="cpu"),
            ]
            self.assertEqual(len(device_result), num_partitions)
            for i in range(num_partitions):
                self.assertEqual(expected_out[i], device_result[i].cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_some_out_empty(self):
        partitions = torch.tensor([0, 0, 2, 2, 0, 2], dtype=torch.int32, device="mlu")
        data = torch.tensor([0, 13, 2, 39, 4, 17], dtype=torch.float32, device="mlu")
        num_partitions = 4
        device_result = torch.ops.torch_mlu.dynamic_partition(
            data, partitions, num_partitions
        )
        # expected out
        expected_out = [
            torch.tensor([0, 13, 4], dtype=torch.float32, device="cpu"),
            torch.tensor([], dtype=torch.float32, device="cpu"),
            torch.tensor([2, 39, 17], dtype=torch.float32, device="cpu"),
            torch.tensor([], dtype=torch.float32, device="cpu"),
        ]
        self.assertEqual(len(device_result), num_partitions)
        for i in range(num_partitions):
            self.assertEqual(expected_out[i], device_result[i].cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_data_zero_numel(self):
        partitions = torch.tensor([0, 0, 2, 2, 0, 2], dtype=torch.int32, device="mlu")
        data = torch.tensor([], dtype=torch.float32, device="mlu").resize_(6, 0)
        num_partitions = 4
        device_result = torch.ops.torch_mlu.dynamic_partition(
            data, partitions, num_partitions
        )
        # expected out
        expected_out = [
            torch.tensor([], dtype=torch.float32, device="cpu").resize_(3, 0),
            torch.tensor([], dtype=torch.float32, device="cpu").resize_(0, 0),
            torch.tensor([], dtype=torch.float32, device="cpu").resize_(3, 0),
            torch.tensor([], dtype=torch.float32, device="cpu").resize_(0, 0),
        ]
        self.assertEqual(len(device_result), num_partitions)
        for i in range(num_partitions):
            self.assertEqual(expected_out[i], device_result[i].cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_partitions_zero_numel(self):
        partitions = torch.tensor([], dtype=torch.int32, device="mlu").resize_(0)
        data = torch.tensor([], dtype=torch.float32, device="mlu").resize_(0, 6)
        num_partitions = 2
        device_result = torch.ops.torch_mlu.dynamic_partition(
            data, partitions, num_partitions
        )
        # expected out
        expected_out = [
            torch.tensor([], dtype=torch.float32, device="cpu").resize_(0, 6),
            torch.tensor([], dtype=torch.float32, device="cpu").resize_(0, 6),
        ]
        self.assertEqual(len(device_result), num_partitions)
        for i in range(num_partitions):
            self.assertEqual(expected_out[i], device_result[i].cpu())


if __name__ == "__main__":
    unittest.main()
