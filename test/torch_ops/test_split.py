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
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestSplitOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_split_and_tensor_split(self):
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
            index_list = []
            index_list.append(random.randint(0, shape[dim] - 1))
            index_list.append(shape[dim] - index_list[0])
            input_mlu = input_t.mlu()
            output_cpu = func(input_t).split(index_list, dim)
            output_mlu = func(input_mlu).split(index_list, dim)
            for index in range(len(output_cpu)):
                self.assertTensorsEqual(
                    output_cpu[index].float(),
                    output_mlu[index].cpu().float(),
                    err,
                    use_MSE=True,
                )
            output_cpu = torch.tensor_split(func(input_t), index_list, dim)
            output_mlu = torch.tensor_split(func(input_mlu), index_list, dim)
            for index in range(len(output_cpu)):
                self.assertTensorsEqual(
                    output_cpu[index].float(),
                    output_mlu[index].cpu().float(),
                    err,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_split_bfloat16(self):
        shape_list = [
            (3, 24, 24),
            (2, 3, 24, 24),
            (2, 10, 3, 24, 24),
            (2, 3, 10, 3, 24, 24),
        ]
        data_types = [
            (torch.bfloat16, 0.0),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func in product(shape_list, data_types, func_list):
            data_type, err = dtype_err
            input_t = torch.randn(shape, dtype=data_type)
            dim = random.randint(0, len(shape) - 1)
            index_list = []
            index_list.append(random.randint(0, shape[dim] - 1))
            index_list.append(shape[dim] - index_list[0])
            input_mlu = input_t.mlu()
            output_cpu = func(input_t).split(index_list, dim)
            output_mlu = func(input_mlu).split(index_list, dim)
            for index in range(len(output_cpu)):
                self.assertTensorsEqual(
                    output_cpu[index].float(),
                    output_mlu[index].cpu().float(),
                    err,
                    use_MSE=True,
                )
            output_cpu = torch.tensor_split(func(input_t), index_list, dim)
            output_mlu = torch.tensor_split(func(input_mlu), index_list, dim)
            for index in range(len(output_cpu)):
                self.assertTensorsEqual(
                    output_cpu[index].float(),
                    output_mlu[index].cpu().float(),
                    err,
                    use_MSE=True,
                )


if __name__ == "__main__":
    unittest.main()
