from __future__ import print_function

import sys
import os
import itertools
import unittest
import logging
from itertools import product
import copy

import torch
import numpy as np

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


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_count_nonzero_dim(self):
        type_list = [True, False]
        shape_list = [
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
            (2, 0, 3),
        ]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len + 1):
                dim_lists = list(itertools.permutations(range(dim_len), i)) + list(
                    itertools.permutations(range(-dim_len, 0), i)
                )
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        out_cpu = x.double().count_nonzero()
                        out_mlu = self.to_device(x).count_nonzero()
                        self.assertTensorsEqual(
                            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_count_nonzero(self):
        shape_list = [
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
            (2, 0, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = x.count_nonzero()
            out_mlu = self.to_mlu(x).count_nonzero()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_count_nonzero_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.count_nonzero(x.double())
        out_mlu = torch.count_nonzero(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_count_nonzero_dtype(self):
        shape = (2, 3, 4)
        type_list = [torch.int, torch.int16, torch.int8, torch.long, torch.uint8]
        out_dtype_list = [
            torch.int8,
            torch.int16,
            torch.int,
            torch.half,
            torch.float,
            torch.double,
        ]
        for t in type_list:
            for out_dtype in out_dtype_list:
                x = (torch.randn(shape, dtype=torch.float) * 10000).to(t)
                out_cpu = x.count_nonzero()
                out_mlu = x.to("mlu").count_nonzero()
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                # TODO(hyl) cnnl cast unsupport int16->int8
                if t is not torch.int16 and out_dtype is not torch.int8:
                    self.assertTensorsEqual(
                        out_cpu.float(),
                        out_mlu.cpu().float(),
                        0.003,
                        use_MSE=True,
                        allow_inf=True,
                    )

        shape_list = [(2, 3, 4), (0)]
        type_list = [
            torch.bool,
            torch.int,
            torch.short,
            torch.int8,
            torch.long,
            torch.uint8,
        ]
        for t in type_list:
            for shape in shape_list:
                x = (torch.randn(shape, dtype=torch.float) * 10000).to(t)
                out_cpu = x.count_nonzero()
                out_mlu = x.to("mlu").count_nonzero()
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
