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


class TestIsfiniteOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_isinf_isnan(self, device="mlu", dtype=torch.float):
        vals = (-float("inf"), float("inf"), float("nan"), -1, 0, 1)
        self.compare_with_numpy(torch.isfinite, numpy.isfinite, vals, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_isinf_isnan_with_cpu(self, dtype=torch.float):
        vals = (-float("inf"), float("inf"), float("nan"), -1, 0, 1)
        x = torch.tensor(vals, dtype=dtype)
        res_cpu = torch.isfinite(x)
        res_mlu = torch.isfinite(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_empty_tensor(self, dtype=torch.float):
        vals = torch.rand([0, 2, 3])
        x = torch.tensor(vals, dtype=dtype)
        res_cpu = torch.isfinite(x)
        res_mlu = torch.isfinite(self.to_mlu(x))
        self.assertTensorsEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_isinf_isnan_int(self, device="mlu", dtype=torch.long):
        vals = (-1, 0, 1)
        self.compare_with_numpy(torch.isfinite, numpy.isfinite, vals, device, dtype)

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_channels_last(self, dtype=torch.float):
        x = torch.randn((3, 4, 5, 6), dtype=dtype).to(memory_format=torch.channels_last)
        x[0][1][2][3] = torch.tensor(-float("inf"), dtype=dtype)
        x[1][2][3][4] = torch.tensor(float("inf"), dtype=dtype)
        x[2][3][4][5] = torch.tensor(float("nan"), dtype=dtype)
        res_cpu = torch.isfinite(x)
        res_mlu = torch.isfinite(self.to_mlu(x))
        self.assertEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_not_dense(self, dtype=torch.float):
        x = torch.randn((3, 4, 5, 6), dtype=dtype).to(memory_format=torch.channels_last)
        x[0][1][2][3] = torch.tensor(-float("inf"), dtype=dtype)
        x[1][2][3][4] = torch.tensor(float("inf"), dtype=dtype)
        x[2][3][4][5] = torch.tensor(float("nan"), dtype=dtype)
        res_cpu = torch.isfinite(x[..., :4])
        res_mlu = torch.isfinite(self.to_mlu(x)[..., :4])
        self.assertEqual(res_cpu, res_mlu.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_isfinite_complex(self):
        type_list = [torch.complex32, torch.complex64]
        for t in type_list:
            print(t)
            x = torch.randn((2, 2, 2, 2), dtype=t)
            x[0][1][0][1] = torch.tensor(complex("inf"), dtype=t)
            x[1][0][1][0] = torch.tensor(complex("nan"), dtype=t)
            x[0][0][1][1] = torch.tensor(-complex("inf"), dtype=t)
            x_mlu = x.mlu()
            res_cpu = x.isfinite()
            res_mlu = x_mlu.isfinite()
            self.assertEqual(res_cpu, res_mlu.cpu())

    # TODO(PYTORCH-10129): cnnlLogic not implement large tensor.
    @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_isfinite_large(self, dtype=torch.float):
        mlu_dtypes = [torch.half]
        for data_type in mlu_dtypes:
            x = torch.randn((5, 1024, 1024, 1024), dtype=dtype)
            x[0][1][2][3] = torch.tensor(-float("inf"), dtype=dtype)
            x[1][2][3][4] = torch.tensor(float("inf"), dtype=dtype)
            x[2][3][4][5] = torch.tensor(float("nan"), dtype=dtype)
            res_cpu = torch.isfinite(x)
            res_mlu = torch.isfinite(self.to_mlu_dtype(x, data_type))
            self.assertEqual(res_cpu, res_mlu.cpu())


if __name__ == "__main__":
    run_tests()
