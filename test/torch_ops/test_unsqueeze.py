from __future__ import print_function

import sys
import os
import unittest
import logging
import copy

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestUnsqueezeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_unsqueeze(self):
        shape_list = [(2, 3, 4, 5, 6), (2, 5), (5, 4, 6), (12, 3, 22, 22)]
        dtype_list = [
            (torch.float, 0),
            (torch.half, 0),
            (torch.int64, 0),
            (torch.int32, 0),
        ]
        for in_shape in shape_list:
            for dim in range(len(in_shape) + 1):
                for data_type, err in dtype_list:
                    input_ = torch.randn(in_shape).to(data_type)
                    output_cpu = torch.unsqueeze(input_, dim)
                    output_mlu = torch.unsqueeze(input_.to("mlu"), dim)
                    self.assertTensorsEqual(
                        output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_unsqueeze_channel_last(self):
        shape_list = [(12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 0)]
        for in_shape in shape_list:
            for dim in range(len(in_shape) + 1):
                for data_type, err in dtype_list:
                    input_ = (
                        torch.randn(in_shape)
                        .to(memory_format=torch.channels_last)
                        .to(data_type)
                    )
                    output_cpu = torch.unsqueeze(input_, dim)
                    output_mlu = torch.unsqueeze(input_.to("mlu"), dim)
                    self.assertTensorsEqual(
                        output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_unsqueeze_not_dense(self):
        shape_list = [(2, 3, 4, 5, 6), (2, 5), (5, 4, 6), (12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 0)]
        for in_shape in shape_list:
            for dim in range(len(in_shape) + 1):
                for data_type, err in dtype_list:
                    input_ = torch.randn(in_shape).to(data_type)
                    output_cpu = torch.unsqueeze(input_[::2], dim)
                    output_mlu = torch.unsqueeze(input_.to("mlu")[::2], dim)
                    self.assertTensorsEqual(
                        output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_unsqueeze_inplace(self):
        shape_list = [(2, 3, 4, 5, 6), (2, 5), (5, 4, 6), (12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 0)]
        for in_shape in shape_list:
            for dim in range(len(in_shape) + 1):
                for data_type, err in dtype_list:
                    input_ = torch.randn(in_shape).to(data_type)
                    input_mlu = self.to_mlu_dtype(copy.deepcopy(input_), data_type)
                    input_mlu_ptr = input_mlu.data_ptr()
                    input_.unsqueeze_(dim)
                    input_mlu.unsqueeze_(dim)
                    input_mlu_ptr_2 = input_mlu.data_ptr()
                    self.assertEqual(input_mlu_ptr, input_mlu_ptr_2)
                    self.assertTensorsEqual(
                        input_.float(), input_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_unsqueeze_inplace_channel_last(self):
        shape_list = [(2, 3, 4, 5), (12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 3e-3)]
        for in_shape in shape_list:
            for dim in range(len(in_shape) + 1):
                for data_type, err in dtype_list:
                    input_ = (
                        torch.randn(in_shape)
                        .to(data_type)
                        .to(memory_format=torch.channels_last)
                    )
                    input_mlu = self.to_mlu_dtype(copy.deepcopy(input_), data_type)
                    input_mlu_ptr = input_mlu.data_ptr()
                    input_.unsqueeze_(dim)
                    input_mlu.unsqueeze_(dim)
                    input_mlu_ptr_2 = input_mlu.data_ptr()
                    self.assertEqual(input_mlu_ptr, input_mlu_ptr_2)
                    self.assertTensorsEqual(
                        input_.float(), input_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_unsqueeze_inplace_not_dense(self):
        shape_list = [(2, 3, 4, 5, 6), (2, 5), (5, 4, 6), (12, 3, 22, 22)]
        dtype_list = [(torch.float, 0), (torch.half, 3e-3)]
        for in_shape in shape_list:
            for dim in range(len(in_shape) + 1):
                for data_type, err in dtype_list:
                    input_ = torch.randn(in_shape).to(data_type)
                    input_mlu = self.to_mlu_dtype(copy.deepcopy(input_), data_type)
                    input_mlu_ptr = input_mlu.data_ptr()
                    input_[::2].unsqueeze_(dim)
                    input_mlu[::2].unsqueeze_(dim)
                    input_mlu_ptr_2 = input_mlu.data_ptr()
                    self.assertEqual(input_mlu_ptr, input_mlu_ptr_2)
                    self.assertTensorsEqual(
                        input_.float(), input_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_unsqueeze_bfloat16(self):
        shape_list = [(2, 3, 4, 5, 6), (2, 5), (5, 4, 6), (12, 3, 22, 22)]
        dtype_list = [
            (torch.bfloat16, 0),
        ]
        for in_shape in shape_list:
            for dim in range(len(in_shape) + 1):
                for data_type, err in dtype_list:
                    input_ = torch.randn(in_shape).to(data_type)
                    output_cpu = torch.unsqueeze(input_, dim)
                    output_mlu = torch.unsqueeze(input_.to("mlu"), dim)
                    self.assertTensorsEqual(
                        output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                    )


if __name__ == "__main__":
    unittest.main()
