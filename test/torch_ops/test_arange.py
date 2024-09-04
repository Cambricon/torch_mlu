from __future__ import print_function

import sys
import os
import math
import unittest
import logging
from itertools import product
import copy

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestArangeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_arange_out(self):
        dtype_list = [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.long,
            torch.float,
            torch.double,
            torch.half,
        ]
        start_list = [0, 3.5, 4.3, 5]
        end_list = [5, 9.5, 60, 120.1]
        step_list = [1, 1.5, 15]
        err = 0.0000001
        contiguous = [True, False]
        for dtype_, start_, end_, step_ in product(
            dtype_list, start_list, end_list, step_list
        ):
            if end_ < start_:
                continue
            len_ = math.ceil((end_ - start_) / step_)
            if dtype_ == torch.long and len_ > 0:
                len_ = math.ceil(((int)(end_) - (int)(start_)) / (int)(step_))

            for con in contiguous:
                if con is False:
                    out_cpu = torch.randn(len_, len_).to(dtype_)
                    out_cpu = torch.as_strided(out_cpu, (len_, len_), (1, 2))
                else:
                    out_cpu = torch.randn(len_).to(dtype_)
                out_mlu = self.to_mlu(copy.deepcopy(out_cpu))
                if dtype_ == torch.half:
                    out_cpu = out_cpu.to(torch.float)
                    err = 0.003

                torch.arange(start_, end_, step_, out=out_cpu)
                torch.arange(start_, end_, step_, device="mlu", out=out_mlu)

                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_arange(self):
        dtype_list = [
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.long,
            torch.float,
            torch.double,
            torch.half,
        ]
        start_list = [0, 3.5, 4.3, 5]
        end_list = [5, 9.5, 60, 120.1]
        step_list = [1, 1.5, 15]
        err = 0.0000001
        for dtype_, start_, end_, step_ in product(
            dtype_list, start_list, end_list, step_list
        ):
            if end_ < start_:
                continue
            len_ = math.ceil((end_ - start_) / step_)
            if dtype_ == torch.long and len_ > 0:
                len_ = math.ceil(((int)(end_) - (int)(start_)) / (int)(step_))

            out_mlu = torch.arange(start_, end_, step_, device="mlu", dtype=dtype_)
            if dtype_ == torch.half:
                dtype_ = torch.float
                err = 0.003
            out_cpu = torch.arange(start_, end_, step_, dtype=dtype_)

            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_arange_bfloat16(self):
        out_cpu = torch.arange(3.5, 11.7, 1.2, device="cpu", dtype=torch.bfloat16)
        out_mlu = torch.arange(3.5, 11.7, 1.2, device="mlu", dtype=torch.bfloat16)
        self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
