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


class TestDynamicStitchOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_simple_1d(self):
        indices = [
            torch.tensor([0, 4, 7], dtype=torch.int32, device="mlu"),
            torch.tensor([1, 6, 2, 3, 5], dtype=torch.int32, device="mlu"),
        ]
        data = [
            torch.tensor([0, 40, 70], dtype=torch.float32, device="mlu"),
            torch.tensor([10, 60, 20, 30, 50], dtype=torch.float32, device="mlu"),
        ]

        # expected out
        expected_out = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70], dtype=torch.float32, device="cpu"
        )
        # Initialize output tensor on the device
        device_result = torch.ops.torch_mlu.dynamic_stitch(indices, data)
        # Move device result to CPU for comparison
        device_result_cpu = device_result.cpu()
        self.assertEqual(expected_out, device_result_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_simple_2d(self):
        dtype_list = [torch.float, torch.half, torch.int, torch.int64]
        for dtype in dtype_list:
            indices = [
                torch.tensor([0, 4, 7], dtype=torch.int32, device="mlu"),
                torch.tensor([1, 6], dtype=torch.int32, device="mlu"),
                torch.tensor([2, 3, 5], dtype=torch.int32, device="mlu"),
            ]
            data = [
                torch.tensor([[0, 1], [40, 41], [70, 71]], dtype=dtype, device="mlu"),
                torch.tensor([[10, 11], [60, 61]], dtype=dtype, device="mlu"),
                torch.tensor([[20, 21], [30, 31], [50, 51]], dtype=dtype, device="mlu"),
            ]

            # expected out
            expected_out = torch.tensor(
                [
                    [0, 1],
                    [10, 11],
                    [20, 21],
                    [30, 31],
                    [40, 41],
                    [50, 51],
                    [60, 61],
                    [70, 71],
                ],
                dtype=dtype,
                device="cpu",
            )
            # Initialize output tensor on the device
            device_result = torch.ops.torch_mlu.dynamic_stitch(indices, data)
            # Move device result to CPU for comparison
            device_result_cpu = device_result.cpu()
            self.assertEqual(expected_out, device_result_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_empty_indices(self):
        indices = [
            torch.tensor([0, 4, 10], dtype=torch.int32, device="mlu"),
            torch.tensor([1, 6, 2, 3, 5], dtype=torch.int32, device="mlu"),
        ]
        data = [
            torch.tensor([0, 40, 100], dtype=torch.float32, device="mlu"),
            torch.tensor([10, 60, 20, 30, 50], dtype=torch.float32, device="mlu"),
        ]

        # expected out
        expected_out = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 0, 0, 0, 100], dtype=torch.float32, device="cpu"
        )
        # Initialize output tensor on the device
        device_result = torch.ops.torch_mlu.dynamic_stitch(indices, data)
        # Move device result to CPU for comparison
        device_result_cpu = device_result.cpu()
        self.assertEqual(expected_out, device_result_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_duplicated_indices(self):
        indices = [
            torch.tensor([0, 4, 7], dtype=torch.int32, device="mlu"),
            torch.tensor([1, 6, 2, 3, 5], dtype=torch.int32, device="mlu"),
            torch.tensor([8, 9, 9, 1, 6], dtype=torch.int32, device="mlu"),
        ]
        data = [
            torch.tensor([0, 40, 70], dtype=torch.float32, device="mlu"),
            torch.tensor([10, 60, 20, 30, 50], dtype=torch.float32, device="mlu"),
            torch.tensor([80, 90, 91, 11, 61], dtype=torch.float32, device="mlu"),
        ]

        # expected out
        expected_out = torch.tensor(
            [0, 11, 20, 30, 40, 50, 61, 70, 80, 91], dtype=torch.float32, device="cpu"
        )
        # Initialize output tensor on the device
        device_result = torch.ops.torch_mlu.dynamic_stitch(indices, data)
        # Move device result to CPU for comparison
        device_result_cpu = device_result.cpu()
        self.assertEqual(expected_out, device_result_cpu)

    # @unittest.skip("not test")
    @testinfo()
    def test_simple_error_IndicesMultiDimensional(self):
        indices = [
            torch.tensor([0, 4, 7], dtype=torch.int32, device="mlu"),
            torch.tensor([[1, 6, 2, 3, 5]], dtype=torch.int32, device="mlu"),
        ]
        data = [
            torch.tensor([0, 40, 70], dtype=torch.float32, device="mlu"),
            torch.tensor([10, 60, 20, 30, 50], dtype=torch.float32, device="mlu"),
        ]

        # Initialize output tensor on the device
        ref_msg = r"data\[1\]\.shape = \[5\] does not start with indices\[1\]\.shape = \[1, 5\]"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            device_result = torch.ops.torch_mlu.dynamic_stitch(indices, data)

    # @unittest.skip("not test")
    @testinfo()
    def test_simple_error_DataDimSizeMismatch(self):
        indices = [
            torch.tensor([0, 4, 5], dtype=torch.int32, device="mlu"),
            torch.tensor([1, 6, 2, 3], dtype=torch.int32, device="mlu"),
        ]
        data = [
            torch.tensor([[0], [40], [70]], dtype=torch.float32, device="mlu"),
            torch.tensor(
                [[10, 11], [60, 61], [20, 21], [30, 31]],
                dtype=torch.float32,
                device="mlu",
            ),
        ]

        # Initialize output tensor on the device
        ref_msg = (
            r"Need data\[0\]\.shape\[1:\] = data\[1\]\.shape\[1:\], got data\[0\]\.shape = \[3, 1\], "
            + r"data\[1\]\.shape = \[4, 2\], indices\[0\]\.shape = \[3\], indices\[1\]\.shape = \[4\]"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            device_result = torch.ops.torch_mlu.dynamic_stitch(indices, data)


if __name__ == "__main__":
    unittest.main()
