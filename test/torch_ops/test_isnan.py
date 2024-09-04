from __future__ import print_function

import sys
import logging
import os
import unittest
import numpy
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestisnanOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_isnan(self, device="mlu", dtype=torch.float):
        vals = (-float("nan"), float("nan"), float("nan"), -1, 0, 1)
        self.compare_with_numpy(torch.isnan, numpy.isnan, vals, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_isnan_with_cpu(self, dtype=torch.float):
        vals = (-float("nan"), float("nan"), float("nan"), -1, 0, 1)
        x = torch.tensor(vals, dtype=dtype)
        res_cpu = torch.isnan(x)
        res_mlu = torch.isnan(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isnan_empty_tensor(self, dtype=torch.float):
        vals = torch.rand([0, 2, 3])
        x = torch.tensor(vals, dtype=dtype)
        res_cpu = torch.isnan(x)
        res_mlu = torch.isnan(self.to_mlu(x))
        self.assertTensorsEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isnan_int(self, device="mlu", dtype=torch.long):
        vals = (-1, 0, 1)
        self.compare_with_numpy(torch.isnan, numpy.isnan, vals, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_isnan_channels_last(self, dtype=torch.float):
        x = torch.randn((3, 4, 5, 6), dtype=dtype).to(memory_format=torch.channels_last)
        x[0][1][2][3] = torch.tensor(-float("nan"), dtype=dtype)
        x[1][2][3][4] = torch.tensor(float("nan"), dtype=dtype)
        x[2][3][4][5] = torch.tensor(float("nan"), dtype=dtype)
        res_cpu = torch.isnan(x)
        res_mlu = torch.isnan(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isnan_not_dense(self, dtype=torch.float):
        x = torch.randn((3, 4, 5, 6), dtype=dtype).to(memory_format=torch.channels_last)
        x[0][1][2][3] = torch.tensor(-float("nan"), dtype=dtype)
        x[1][2][3][4] = torch.tensor(float("nan"), dtype=dtype)
        x[2][3][4][5] = torch.tensor(float("nan"), dtype=dtype)
        res_cpu = torch.isnan(x[..., :4])
        res_mlu = torch.isnan(self.to_mlu(x)[..., :4])
        self.assertEqual(res_cpu, res_mlu.cpu())

    # TODO(PYTORCH-10129): cnnlLogic not implement large tensor.
    @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("44GB")
    def test_isnan_large(self):
        dtype = torch.float
        shape = (5, 1024, 1024, 1024)
        x = torch.randn(shape, dtype=dtype)
        x[0][1][2][3] = torch.tensor(-float("nan"), dtype=dtype)
        x[1][2][3][4] = torch.tensor(float("nan"), dtype=dtype)
        x[2][3][4][5] = torch.tensor(float("nan"), dtype=dtype)
        res_cpu = torch.isnan(x)
        res_mlu = torch.isnan(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())


if __name__ == "__main__":
    run_tests()
