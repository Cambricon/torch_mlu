from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestSetOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_set(self):
        in_shape1 = (2, 3)
        in_shape2 = (3, 4)
        input_dtypes = [torch.float, torch.half]
        for data_type in input_dtypes:
            input1 = torch.rand(in_shape1).to(data_type)
            input2 = torch.rand(in_shape2).to(data_type)
            input1_mlu = input1.to("mlu")
            input2_mlu = input2.to("mlu")
            input1.set_(input2)
            input1_mlu.set_(input2_mlu)
            self.assertTensorsEqual(
                input1.float(), input1_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTensorsEqual(
                input1_mlu.cpu().float(), input2_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(input1.size(), input1_mlu.size())
            self.assertTrue(input1.stride(), input1_mlu.stride())
            input3 = torch.rand(in_shape1).to(data_type)
            input3_mlu = input3.mlu()
            input3.set_()
            input3_mlu.set_()
            self.assertTensorsEqual(
                input3.float(), input3_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(input3.size(), input3_mlu.size())
            self.assertTrue(input3.stride(), input3_mlu.stride())

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_set_bfloat16(self):
        in_shape1 = (2, 3)
        in_shape2 = (3, 4)
        data_type = torch.bfloat16
        input1 = torch.rand(in_shape1).to(data_type)
        input2 = torch.rand(in_shape2).to(data_type)
        input1_mlu = input1.to("mlu")
        input2_mlu = input2.to("mlu")
        input1.set_(input2)
        input1_mlu.set_(input2_mlu)
        self.assertTensorsEqual(
            input1.float(), input1_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            input1_mlu.cpu().float(), input2_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTrue(input1.size(), input1_mlu.size())
        self.assertTrue(input1.stride(), input1_mlu.stride())
        input3 = torch.rand(in_shape1).to(data_type)
        input3_mlu = input3.mlu()
        input3.set_()
        input3_mlu.set_()
        self.assertTensorsEqual(
            input3.float(), input3_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTrue(input3.size(), input3_mlu.size())
        self.assertTrue(input3.stride(), input3_mlu.stride())


if __name__ == "__main__":
    unittest.main()
