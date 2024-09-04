from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestStackOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_stack(self):
        # The dimensions of the input tensors must be equal
        shape_list = [(3, 5), (5, 6, 7), (1, 2, 3, 4), (5, 2, 30, 4, 10)]
        type_list = [
            torch.float,
            torch.int8,
            torch.int16,
            torch.int64,
            torch.long,
            torch.bool,
        ]
        channel_first = [True, False]
        for shape in shape_list:
            for type in type_list:
                for dim in [-len(shape) - 1, len(shape)]:
                    for channel in channel_first:
                        a_1 = torch.ones(shape, dtype=type)
                        a_2 = torch.ones(shape, dtype=type)
                        if channel is False:
                            a_1 = self.convert_to_channel_last(a_1)
                            a_2 = self.convert_to_channel_last(a_2)
                        out_cpu = torch.stack((a_1, a_2), dim=dim)
                        out_mlu = torch.stack((a_1.to("mlu"), a_2.to("mlu")), dim=dim)
                        self.assertTensorsEqual(out_cpu, out_mlu.cpu().contiguous(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_stack_not_dense(self):
        # The dimensions of the input tensors must be equal
        shape_list = [(3, 5), (5, 6, 7), (1, 2, 3, 4), (5, 2, 30, 4, 10)]
        type_list = [
            torch.float,
            torch.int8,
            torch.int16,
            torch.int64,
            torch.long,
            torch.bool,
        ]
        for shape in shape_list:
            for type in type_list:
                for dim in [-len(shape) - 1, len(shape)]:
                    a_1 = torch.ones(shape, dtype=type)
                    a_2 = torch.ones(shape, dtype=type)
                    out_cpu = torch.stack((a_1[::2], a_2[::2]), dim=dim)
                    out_mlu = torch.stack(
                        (a_1.to("mlu")[::2], a_2.to("mlu")[::2]), dim=dim
                    )
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu().contiguous(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_stack_out(self):
        # The dimensions of the input tensors must be equal
        shape_list = [(3, 5), (5, 6, 7), (1, 2, 3, 4), (5, 2, 30, 4, 10)]
        type_list = [
            torch.float,
            torch.int8,
            torch.int16,
            torch.int64,
            torch.long,
            torch.bool,
        ]
        channel_first = [True, False]
        for shape in shape_list:
            for type in type_list:
                for dim in [-len(shape) - 1, len(shape)]:
                    for channel in channel_first:
                        a_1 = torch.ones(shape, dtype=type)
                        a_2 = torch.ones(shape, dtype=type)
                        if channel is False:
                            a_1 = self.convert_to_channel_last(a_1)
                            a_2 = self.convert_to_channel_last(a_2)
                        out_cpu = torch.ones((4,), dtype=type)
                        out_mlu = out_cpu.to("mlu")
                        torch.stack((a_1, a_2), dim=dim, out=out_cpu)
                        torch.stack(
                            (a_1.to("mlu"), a_2.to("mlu")), dim=dim, out=out_mlu
                        )
                        self.assertTensorsEqual(out_cpu, out_mlu.cpu().contiguous(), 0)


if __name__ == "__main__":
    unittest.main()
