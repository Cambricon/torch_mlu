from __future__ import print_function
import sys
import os
import unittest
import logging
from itertools import product
import copy

import numpy
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    TestCase,
    testinfo,
    run_tests,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413
import torch_mlu  # pylint: disable=W0611, C0413

logging.basicConfig(level=logging.DEBUG)


class TestHistcOp(TestCase):
    # TODO(xujian1) temp not test
    # @unittest.skip("not test")
    @testinfo()
    def test_histc(self):
        dtype_list = [  # (torch.int8, 3e-3),
            (torch.uint8, 3e-3),
            (torch.int16, 3e-3),
            (torch.int32, 3e-3),
            (torch.int64, 3e-3),
            (torch.float, 3e-3),
            (torch.double, 3e-3),
        ]
        shape_list = [(10, 12, 10, 13), (2, 10, 15)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        for dtype_err, in_shape, func in product(dtype_list, shape_list, func_list):
            dtype, err = dtype_err
            x_cpu = torch.randint(4, 100, in_shape).to(dtype)
            x_mlu = x_cpu.to("mlu")
            out_cpu = torch.histc(func(x_cpu.float()))
            out_mlu = torch.histc(func(x_mlu))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

        # test sliced output
        y_cpu = torch.randn(10, 25)
        y_mlu = y_cpu.to("mlu")
        out_cpu = torch.randn(10, 50)
        out_mlu = out_cpu.to("mlu")
        torch.histc(y_cpu, out=out_cpu[:, :2])
        torch.histc(y_mlu, out=out_mlu[:, :2])
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_histc_orig(self):
        device = "mlu"
        # empty tensor
        actual = torch.histc(torch.tensor([], device=device), min=0, max=3)
        expected = torch.zeros(100, dtype=torch.float, device=device)
        self.assertEqual(expected, actual)

        # without nbins
        actual = torch.histc(torch.tensor([2, 5], dtype=torch.float, device=device))
        expected = torch.zeros(100, dtype=torch.float, device=device)
        expected[0] = 1
        expected[99] = 1
        self.assertEqual(expected, actual)
        # tensor with the same element
        actual = torch.histc(torch.ones(5, dtype=torch.float, device=device), bins=5)
        self.assertEqual(
            torch.tensor([0, 0, 5, 0, 0], dtype=torch.float, device=device), actual
        )
        # no element falls between [min, max]
        actual = torch.histc(
            torch.ones(5, dtype=torch.float, device=device), bins=5, min=2, max=3
        )
        self.assertEqual(
            torch.tensor([0, 0, 0, 0, 0], dtype=torch.float, device=device), actual
        )
        # element falls below min + integral bin size and
        actual = torch.histc(
            torch.tensor([2, 4, 2, 2, 5, 4], dtype=torch.float, device=device),
            bins=5,
            min=1,
            max=5,
        )
        self.assertEqual(
            torch.tensor([0, 3, 0, 2, 1], dtype=torch.float, device=device), actual
        )
        # non-integral bin size
        actual = torch.histc(
            torch.tensor([1, 2, 1], dtype=torch.float, device=device),
            bins=4,
            min=0,
            max=3,
        )
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.float, device=device), actual
        )
        # double input
        actual = torch.histc(
            torch.tensor([1, 2, 1], dtype=torch.double, device=device),
            bins=4,
            min=0,
            max=3,
        )
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.double, device=device), actual
        )
        self.assertEqual(actual.dtype, torch.double)
        # mixed input
        actual = torch.histc(
            torch.tensor([1.0, 2, 1], dtype=torch.float, device=device),
            bins=4,
            min=0,
            max=3,
        )
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.float, device=device), actual
        )
        self.assertEqual(actual.dtype, torch.float)
        # scalar input and 1 bin -- should return a 1-dimensional tensor, not a scalar.
        actual = torch.histc(
            torch.tensor(0, dtype=torch.float, device=device), bins=1, min=0, max=3
        )
        self.assertEqual(torch.tensor([1], dtype=torch.float, device=device), actual)
        # tensors with inf; min, max provided
        self.assertEqual(
            torch.histc(
                torch.tensor([float("inf")], dtype=torch.float, device=device),
                bins=1,
                min=0,
                max=3,
            ),
            torch.tensor([0], dtype=torch.float, device=device),
        )
        self.assertEqual(
            torch.histc(
                torch.tensor(
                    [1.0, 2.0, float("inf")], dtype=torch.float, device=device
                ),
                bins=4,
                max=3,
            ),
            torch.tensor([0, 1, 1, 0], dtype=torch.float, device=device),
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_histc_exception(self):
        device = "mlu"
        # negative nbins throws
        with self.assertRaisesRegex(RuntimeError, "bins must be > 0"):
            torch.histc(torch.tensor([1], dtype=torch.float, device=device), bins=-1)
        # tensors with min > max -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, "max must be larger than min"):
            torch.histc(
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float, device=device),
                bins=4,
                min=5,
                max=1,
            )
        # tensors with inf; min, max not provided -- should throw a RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, r"range of \[inf, inf\] is not finite"
        ):
            torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r"range of \[1, inf\] is not finite"):
            torch.histc(
                torch.tensor([1.0, 2.0, float("inf")], dtype=torch.float, device=device)
            )
        # tensor with nan -- should throw a RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, r"range of \[nan, nan\] is not finite"
        ):
            torch.histc(torch.tensor([float("nan")], dtype=torch.float, device=device))
        ref_msg = "torch.histogram: input tensor and hist tensor should have the same dtype,\
 but got input float and hist int"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
            input = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
            torch.histc(input, out=out)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("21GB")
    def test_histc_large(self):
        in_shape = (4, 1025, 1024, 1024)
        # TODO: Large tensor is not supported when dtype of input is float32
        dtype, err = torch.int, 3e-3
        x_cpu = torch.randint(4, 100, in_shape).to(dtype)
        x_cpu_np = x_cpu.numpy()
        x_mlu = x_cpu.to("mlu")
        out_cpu_np = numpy.histogram(x_cpu_np, bins=100)
        out_cpu = torch.from_numpy(out_cpu_np[0])
        out_mlu = torch.histc(x_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), err, use_MSE=True)


if __name__ == "__main__":
    run_tests()
