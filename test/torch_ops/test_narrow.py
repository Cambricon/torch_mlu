from __future__ import print_function
import logging
import sys
import os
import unittest
import random

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411


class TestNarrowOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_narrow(self):
        shape_list = [(2, 3, 4, 5, 6), (2, 5), (5, 4, 6), (12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 3e-3)]
        for in_shape in shape_list:
            for dim in range(-len(in_shape), len(in_shape) - 1):
                for data_type, err in dtype_list:
                    start = random.randint(0, in_shape[dim])
                    length = random.randint(0, in_shape[dim] - 1)
                    if length > in_shape[dim] - start:
                        length = in_shape[dim] - start
                    input_ = torch.randn(in_shape, dtype=torch.float)
                    output_cpu = torch.narrow(input_, dim, start, length)
                    output_mlu = torch.narrow(
                        self.to_mlu_dtype(input_, data_type), dim, start, length
                    )
                    self.assertTensorsEqual(
                        output_cpu, output_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_narrow_channels_last(self):
        shape = (12, 3, 22, 22)
        x = torch.randn(shape).to(memory_format=torch.channels_last)
        x_mlu = self.to_mlu(x)
        output_cpu = torch.narrow(x, 2, 0, 11)
        output_mlu = torch.narrow(x_mlu, 2, 0, 11)
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_narrow_no_dense(self):
        shape = (12, 3, 22, 22)
        x = torch.randn(shape)
        x_mlu = self.to_mlu(x)[..., ::2]
        x = x[..., ::2]
        output_cpu = torch.narrow(x, 2, 0, 5)
        output_mlu = torch.narrow(x_mlu, 2, 0, 5)
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_narrow_copy(self):
        shape = (12, 3, 22, 22)
        x = torch.randn(shape)
        x_mlu = self.to_mlu(x)[..., ::2]
        x = x[..., ::2]
        output_cpu = x.narrow_copy(2, 0, 5)
        output_mlu = x_mlu.narrow_copy(2, 0, 5)
        self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), 0.0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
