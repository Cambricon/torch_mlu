from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestUnbindOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_unbind(self):
        dtype_list = [
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.int16,
            torch.int32,
        ]
        shape_list = [(99,), (12, 24), (3, 18, 9), (15, 25, 35, 2), (5, 16, 9, 10, 2)]
        for shape, dtype in product(shape_list, dtype_list):
            for dim in range(-len(shape), len(shape)):
                input_cpu = torch.randn(shape).to(dtype)
                input_mlu = input_cpu.to("mlu")
                output_cpu = torch.unbind(input_cpu, dim)
                output_mlu = torch.unbind(input_mlu, dim)
                for oc, om in zip(output_cpu, output_mlu):
                    self.assertTensorsEqual(oc.float(), om.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_unbind_channel_last(self):
        dtype_list = [
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.int16,
            torch.int32,
        ]
        shape_list = [(15, 25, 35, 2)]
        for shape, dtype in product(shape_list, dtype_list):
            for dim in range(-len(shape), len(shape)):
                input_cpu = (
                    torch.randn(shape).to(dtype).to(memory_format=torch.channels_last)
                )
                input_mlu = input_cpu.to("mlu")
                output_cpu = torch.unbind(input_cpu, dim)
                output_mlu = torch.unbind(input_mlu, dim)
                for oc, om in zip(output_cpu, output_mlu):
                    self.assertTensorsEqual(oc.float(), om.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_unbind_not_dense(self):
        dtype_list = [
            torch.float,
            torch.half,
            torch.double,
            torch.int8,
            torch.int16,
            torch.int32,
        ]
        shape_list = [(12, 24), (3, 18, 9), (15, 25, 35, 2), (5, 16, 9, 10, 2)]
        for shape, dtype in product(shape_list, dtype_list):
            for dim in range(-len(shape), len(shape)):
                input_cpu = torch.randn(shape).to(dtype)
                input_mlu = input_cpu.to("mlu")
                output_cpu = torch.unbind(input_cpu[::2], dim)
                output_mlu = torch.unbind(input_mlu[::2], dim)
                for oc, om in zip(output_cpu, output_mlu):
                    self.assertTensorsEqual(oc.float(), om.cpu().float(), 0)


if __name__ == "__main__":
    unittest.main()
