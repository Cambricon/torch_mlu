from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import TestCase, testinfo  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_ones(self):
        # test dim 0
        result_cpu = torch.ones(0)
        result_mlu = torch.ones(0, device="mlu")
        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

        dtype_list = [
            torch.float,
            torch.half,
            torch.int16,
            torch.int8,
            torch.int32,
            torch.double,
        ]
        shape_list = [(10), (932, 9777), (5, 6, 7), (3, 4, 4, 3)]
        for shape, dtype in product(shape_list, dtype_list):
            result_cpu = torch.ones(shape).to(dtype)
            result_mlu = torch.ones(shape, device="mlu")
            self.assertTensorsEqual(result_cpu.float(), result_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_ones_like_contiguous(self):
        shape_list = [(10), (932, 9777), (5, 6, 7), (3, 4, 4, 3)]
        for shape in shape_list:
            a = torch.rand(shape, dtype=torch.float)
            result_cpu = torch.ones_like(a)
            result_mlu = torch.ones_like(a.mlu())
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_ones_like_channel_last(self):
        shape_list = [(3, 4, 4, 3), (2, 3, 3, 32, 32)]
        for shape in shape_list:
            a = torch.rand(shape, dtype=torch.float)
            a = self.convert_to_channel_last(a)
            result_cpu = torch.ones_like(a)
            result_mlu = torch.ones_like(a.mlu())
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_ones_not_dense(self):
        shape_list = [(932, 9777 * 2), (5, 6, 7 * 2), (3, 4, 4, 3 * 2)]
        for shape in shape_list:
            a = torch.empty(0)
            if len(shape) == 2:
                a = torch.rand(shape, dtype=torch.float)[:, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                a = torch.rand(shape, dtype=torch.float)[:, :, : int(shape[-1] / 2)]
            elif len(shape) == 4:
                a = torch.rand(shape, dtype=torch.float)[:, :, :, : int(shape[-1] / 2)]
            result_cpu = torch.ones_like(a)
            result_mlu = torch.ones_like(a.mlu())
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_ones_with_out(self):
        shape_list = [(10), (932, 9777), (5, 6, 7), (3, 4, 4, 3)]
        for shape in shape_list:
            result_cpu = torch.empty(shape)
            result_mlu = result_cpu.to("mlu")
            torch.ones(shape, out=result_cpu)
            torch.ones(shape, out=result_mlu)
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_new_ones(self):
        size_list = [
            (),
            (2,),
            (2, 3, 4),
            (2, 3, 4, 5),
            (2, 0, 3, 4),
            (2, 3, 4, 5, 6),
            (2, 3, 4, 5, 6, 7),
        ]
        type_list = [
            None,
            torch.bool,
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.int8,
            torch.long,
            torch.uint8,
        ]
        data = torch.tensor(())
        for size, type in product(size_list, type_list):
            result_cpu = data.new_ones(size, dtype=type)
            result_mlu = data.new_ones(size, dtype=type, device="mlu")
            self.assertTensorsEqual(
                result_cpu,
                result_mlu.cpu() if type != torch.half else result_mlu.cpu().float(),
                0,
            )
            self.assertEqual(result_mlu.dtype, result_cpu.dtype)
            self.assertEqual(result_mlu.dtype, type if type else torch.float)
            self.assertEqual(result_mlu.size(), result_cpu.size())
            self.assertEqual(result_mlu.size(), size)


if __name__ == "__main__":
    unittest.main()
