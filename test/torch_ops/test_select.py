from __future__ import print_function

import sys
import os
import unittest
import logging
import random
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestSelectOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_select(self):
        shape_list = [
            (3, 24, 24),
            (2, 3, 24, 24),
            (2, 10, 3, 24, 24),
            (2, 3, 10, 3, 24, 24),
        ]
        data_types = [(torch.float, 0.0), (torch.half, 0.0)]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func in product(shape_list, data_types, func_list):
            data_type, err = dtype_err
            input_t = torch.randn(shape, dtype=data_type)
            dim = random.randint(0, len(shape) - 1)
            index = random.randint(-shape[dim], shape[dim] - 1)
            input_mlu = input_t.mlu()
            output_cpu = func(input_t).select(dim, index)
            output_mlu = func(input_mlu).select(dim, index)
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
            )


if __name__ == "__main__":
    unittest.main()
