from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    read_card_info,
    TestCase,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestPermuteOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_permute(self):
        shape_permute_list = [
            [(3, 224, 224), (0, 2, 1)],
            [(2, 3, 224, 224), (0, 3, 1, 2)],
            [(2, 3, 224, 224), (-4, -1, -3, -2)],
            [(2, 3, 224, 224), (0, 1, 2, 3)],
            [(2, 10, 3, 224, 224), (0, 4, 1, 2, 3)],
            [(2, 10, 3, 224, 224), (-5, -1, -4, -3, -2)],
            [(2, 3, 10, 3, 224, 224), (0, 4, 5, 1, 2, 3)],
            [(2, 3, 10, 3, 224, 224), (-6, -2, -1, -5, -4, -3)],
            [(2, 10, 3, 224, 224), (0, -1, 1, 2, -2)],
            [(2, 3, 10, 3, 224, 224), (0, 4, 5, 1, 2, 3)],
        ]
        data_types = [(torch.float, 0.0), (torch.half, 0.0)]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape_permute, dtype_err, func in product(
            shape_permute_list, data_types, func_list
        ):
            data_type, err = dtype_err
            shape, permute_index = shape_permute
            input_t = torch.randn(shape, dtype=data_type)
            input_mlu = input_t.mlu()
            output_cpu = func(input_t).permute(permute_index)
            output_mlu = func(input_mlu).permute(permute_index)
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_permute_not_dense_and_channel_last(self):
        shape_permute = [
            [(3, 224, 224), (0, 2, 1)],
            [(2, 3, 224, 224), (0, 3, 1, 2)],
            [(2, 3, 224, 224), (-4, -1, -3, -2)],
            [(2, 3, 224, 224), (0, 1, 2, 3)],
            [(2, 10, 3, 224, 224), (0, 4, 1, 2, 3)],
            [(2, 10, 3, 224, 224), (-5, -1, -4, -3, -2)],
            [(2, 3, 10, 3, 224, 224), (-6, -2, -1, -5, -4, -3)],
            [(2, 10, 3, 224, 224), (0, -1, 1, 2, -2)],
            [(2, 3, 10, 3, 224, 224), (0, 4, 5, 1, 2, 3)],
        ]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for data_type, err in data_types:
            for shape, permute_index in shape_permute:
                if len(shape) == 4:
                    memory_type = torch.channels_last
                elif len(shape) == 5:
                    memory_type = torch.channels_last_3d
                else:
                    memory_type = torch.contiguous_format
                input_t = torch.rand(shape).to(memory_format=memory_type)
                input_mlu = self.to_mlu_dtype(input_t, data_type)
                output_cpu = input_t[..., :112].permute(permute_index)
                output_mlu = input_mlu[..., :112].permute(permute_index)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), err, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_permute_bfloat16(self):
        input_t = torch.rand((2, 2, 24, 3), dtype=torch.bfloat16)
        input_cpu = torch.nn.Parameter(input_t)
        input_mlu = torch.nn.Parameter(input_t.mlu())
        out_cpu = input_cpu.permute((1, 3, 2, 0))
        out_mlu = input_mlu.permute((1, 3, 2, 0))
        grad = torch.rand_like(out_cpu)
        out_cpu.backward(grad)
        out_mlu.backward(grad.mlu())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=True)
        self.assertTensorsEqual(input_cpu.grad, input_mlu.grad.cpu(), 0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
