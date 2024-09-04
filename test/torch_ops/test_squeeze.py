from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestSqueezeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0.0)]
        for in_shape in [(2, 1, 2, 1, 2), (2, 3, 4)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape).to(data_type)
                output_cpu = torch.squeeze(input1)
                output_mlu = torch.squeeze(input1.to("mlu"))
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_channel_last(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0)]
        for in_shape in [(2, 1, 2, 1), (2, 3, 4, 5)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape).to(data_type)
                input1_mlu = input1.to("mlu").to(memory_format=torch.channels_last)
                output_cpu = torch.squeeze(input1.to(memory_format=torch.channels_last))
                output_mlu = torch.squeeze(input1_mlu)
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_not_dense(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0)]
        for in_shape in [(4, 5, 1, 3, 4), (2, 3, 4)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape).to(data_type)
                output_cpu = torch.squeeze(input1[::2])
                output_mlu = torch.squeeze(input1.to("mlu")[::2])
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_inplace(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0)]
        for in_shape in [(2, 1, 2, 1, 2)]:
            for dim in [1, 3]:
                for data_type, err in dtype_list:
                    input_t = torch.randn(in_shape).to(data_type)
                    input_mlu = copy.deepcopy(input_t).to("mlu")
                    input_t.squeeze_(dim)
                    input_mlu.squeeze_(dim)
                    self.assertTensorsEqual(
                        input_t.float(), input_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_inplace_channel_last(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0)]
        for in_shape in [(4, 5, 3, 4)]:
            for dim in [1, 3]:
                for data_type, err in dtype_list:
                    input = torch.randn(in_shape).to(data_type)
                    input_t = input.to(memory_format=torch.channels_last)
                    input_mlu = input.to("mlu").to(memory_format=torch.channels_last)
                    input_t.squeeze_(dim)
                    input_mlu.squeeze_(dim)
                    self.assertTensorsEqual(
                        input_t.float(), input_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_squeeze_inplace_not_dense(self):
        dtype_list = [(torch.float, 0.0), (torch.half, 0)]
        for in_shape in [(4, 5, 1, 3, 4)]:
            for dim in [1, 3]:
                for data_type, err in dtype_list:
                    input_t = torch.randn(in_shape).to(data_type)
                    input_mlu = copy.deepcopy(input_t).to("mlu")
                    input_t[::2].squeeze_(dim)
                    input_mlu[::2].squeeze_(dim)
                    self.assertTensorsEqual(
                        input_t.float(), input_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_squeeze_bfloat16(self):
        dtype_list = [
            (torch.bfloat16, 0.0),
        ]
        for in_shape in [(2, 1, 2, 1, 2), (2, 3, 4)]:
            for data_type, err in dtype_list:
                input1 = torch.randn(in_shape).to(data_type)
                output_cpu = torch.squeeze(input1)
                output_mlu = torch.squeeze(input1.to("mlu"))
                self.assertEqual(output_cpu.size(), output_mlu.size())
                self.assertEqual(output_cpu.stride(), output_mlu.stride())
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                )


if __name__ == "__main__":
    unittest.main()
