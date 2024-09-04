from __future__ import print_function
import logging
import sys
import os
import unittest

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411


class TestChunkOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_chunk(self):
        shape_list = [(2, 3, 4, 5, 6), (2, 5), (5, 4, 6), (12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 3e-3)]
        chunks_list = [2, 3, 4]
        for in_shape in shape_list:
            for dim in range(-len(in_shape), len(in_shape) - 1):
                for data_type, err in dtype_list:
                    for chunk in chunks_list:
                        input_ = torch.randn(in_shape, dtype=torch.float)
                        output_cpu = torch.chunk(input_, chunk, dim)
                        output_mlu = torch.chunk(
                            self.to_mlu_dtype(input_, data_type), chunk, dim
                        )
                        for index, elem in enumerate(output_cpu):
                            self.assertTensorsEqual(
                                elem, output_mlu[index].cpu().float(), err, use_MSE=True
                            )

    # @unittest.skip("not test")
    @testinfo()
    def test_chunk_not_dense(self):
        shape_list = [(5, 4, 6, 8), (12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 3e-3)]
        chunks_list = [2, 3, 4]
        for in_shape in shape_list:
            for dim in range(-len(in_shape), len(in_shape) - 1):
                for data_type, err in dtype_list:
                    for chunk in chunks_list:
                        input_ = torch.randn(in_shape, dtype=torch.float)
                        output_cpu = torch.chunk(input_[:, :, :, 1:3], chunk, dim)
                        input_mlu = self.to_mlu_dtype(input_, data_type)
                        output_mlu = torch.chunk(input_mlu[:, :, :, 1:3], chunk, dim)
                        for index, elem in enumerate(output_cpu):
                            self.assertTensorsEqual(
                                elem, output_mlu[index].cpu().float(), err, use_MSE=True
                            )

    # @unittest.skip("not test")
    @testinfo()
    def test_chunk_channel_last(self):
        shape_list = [(2, 3, 5, 6), (12, 3, 22, 22, 16)]
        dtype_list = [(torch.float, 0)]
        chunks_list = [2, 3, 4]
        for in_shape in shape_list:
            for dim in range(-len(in_shape), len(in_shape) - 1):
                for data_type, err in dtype_list:
                    for chunk in chunks_list:
                        input = torch.randn(in_shape, dtype=torch.float)
                        input_ = self.convert_to_channel_last(input)
                        output_cpu = torch.chunk(input_, chunk, dim)
                        input_mlu = self.to_mlu_dtype(input, data_type)
                        input_mlu = self.convert_to_channel_last(input_mlu)
                        output_mlu = torch.chunk(input_mlu, chunk, dim)
                        for index, elem in enumerate(output_cpu):
                            self.assertTensorsEqual(
                                elem, output_mlu[index].cpu().float(), err, use_MSE=True
                            )


if __name__ == "__main__":
    unittest.main()
