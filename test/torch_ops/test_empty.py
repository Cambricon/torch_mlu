from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413

logging.basicConfig(level=logging.DEBUG)


class TestEmpty(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_empty(self):
        shape_memory_format_list = [
            ((2, 3, 4, 5), torch.channels_last, False),
            ((6, 7, 8, 9), torch.contiguous_format, False),
            ((5, 4, 3, 2), torch.contiguous_format, True),
            ((64, 3, 224, 224), torch.channels_last, True),
            ((5, 0, 3, 2), torch.contiguous_format, False),
            ((2, 3, 0, 5), torch.channels_last, False),
        ]
        for shape, memory_format, pin_memory in shape_memory_format_list:
            x_mlu = torch.empty(shape, memory_format=memory_format, device="mlu")
            x_mlu_to_cpu = torch.empty(
                shape, memory_format=memory_format, device="mlu"
            ).cpu()
            x_cpu = torch.empty(
                shape, memory_format=memory_format, pin_memory=pin_memory
            )
            self.assertEqual(x_cpu.size(), x_mlu.size())
            self.assertEqual(x_cpu.size(), x_mlu_to_cpu.size())
            self.assertEqual(x_cpu.stride(), x_mlu.stride())
            self.assertEqual(x_cpu.stride(), x_mlu_to_cpu.stride())
            if pin_memory is True:
                self.assertTrue(
                    x_cpu.is_pinned(), "create cpu pin memory tensor failed."
                )


if __name__ == "__main__":
    unittest.main()
