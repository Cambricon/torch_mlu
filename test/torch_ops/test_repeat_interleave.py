from __future__ import print_function
from copy import copy
from itertools import product

import sys
import os
import unittest
import logging
import copy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_interleave(self):
        shape_list = [
            (2, 3, 4, 5),
            (1, 2, 2),
            (2, 2),
            (3,),
            (5, 6, 10, 12, 12),
            (),
            (1024, 1024),
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        dtypes = [torch.double, torch.float, torch.long, torch.int]
        for shape, func in product(shape_list, func_list):
            for t in dtypes:
                x = torch.randint(-16777216, 16777216, shape).to(t)
                x_cpu = func(x)
                output_cpu = x_cpu.repeat_interleave(2)
                output_mlu = self.to_mlu(x_cpu).repeat_interleave(2)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-4, use_MSE=True
                )

        # test no self
        repeats = torch.tensor([1, 4, 3])
        repeats_mlu = self.to_mlu(repeats)
        output_cpu = torch.repeat_interleave(repeats)
        output_mlu = torch.repeat_interleave(repeats_mlu)
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 3e-4, use_MSE=True
        )

        y = torch.tensor([[1, 2], [3, 4]])
        y_mlu = self.to_mlu(y)

        # test dim
        y1_cpu = torch.repeat_interleave(y, 3, dim=1)
        y1_mlu = torch.repeat_interleave(y_mlu, 3, dim=1)
        self.assertTensorsEqual(
            y1_cpu.float(), y1_mlu.cpu().float(), 3e-4, use_MSE=True
        )

        y2_cpu = torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
        y2_mlu = torch.repeat_interleave(
            y_mlu, self.to_mlu(torch.tensor([1, 2])), dim=0
        )
        self.assertTensorsEqual(
            y2_cpu.float(), y2_mlu.cpu().float(), 3e-4, use_MSE=True
        )

        # test output_size
        repeats_out_size_list = [
            (torch.tensor([1, 2]), 3),
            (torch.tensor([1, 4]), 5),
            (torch.tensor([2, 4]), 6),
            (torch.tensor([3, 5]), 8),
        ]
        for repeats, output_size in repeats_out_size_list:
            y3_cpu = torch.repeat_interleave(y, repeats, dim=0, output_size=output_size)
            y3_mlu = torch.repeat_interleave(
                y_mlu, repeats.to("mlu"), dim=0, output_size=output_size
            )
            self.assertTensorsEqual(
                y3_cpu.float(), y3_mlu.cpu().float(), 3e-4, use_MSE=True
            )

        # test zero sized dimension
        x = torch.zeros((5, 0))
        x_mlu = self.to_mlu(x)
        y_cpu = torch.repeat_interleave(x, repeats=3, dim=1)
        y_mlu = torch.repeat_interleave(x_mlu, repeats=3, dim=1)
        self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 3e-4, use_MSE=True)

        x = torch.tensor([], dtype=torch.int64)
        x_mlu = self.to_mlu(x)
        y_cpu = torch.repeat_interleave(x, x)
        y_mlu = torch.repeat_interleave(x_mlu, x_mlu)
        self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 3e-4, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_interleave_shape_limit(self):
        input1 = torch.randint(-16777216, 16777216, (250, 400))
        channels_last_input1 = self.convert_to_channel_last(input1)
        repeats = torch.randint(0, 10, (100000,)).to(torch.long)
        output_cpu = channels_last_input1.repeat_interleave(repeats)
        output_mlu = self.to_mlu(channels_last_input1).repeat_interleave(
            repeats.to("mlu")
        )
        output_mlu_channels_first = output_mlu.cpu().contiguous()
        self.assertTensorsEqual(
            output_cpu, output_mlu_channels_first, 3e-4, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_interleave_backward(self):
        x = torch.randn((2, 2), requires_grad=True)
        x_mlu = copy.deepcopy(x)

        out_cpu = torch.repeat_interleave(x, 2)
        out_mlu = torch.repeat_interleave(x_mlu.to("mlu"), 2)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-4, use_MSE=True
        )

        grad = torch.randn(out_cpu.shape)
        out_cpu.backward(grad)
        out_mlu.backward(grad.to("mlu"))
        self.assertTensorsEqual(x.grad, x_mlu.grad.cpu(), 3e-4, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_repeat_interleave_exception(self):
        ref_msg = "repeat_interleave only accept 1D vector as repeat"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.repeat_interleave(torch.tensor([[1, 2], [3, 4]]).to("mlu"))

        ref_msg = "repeats has to be Long or Int tensor"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.repeat_interleave(torch.Tensor([1, 2]).to("mlu"))

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_repeat_interleave_bfloat16(self):
        x = torch.randn((2, 2), requires_grad=True, dtype=torch.bfloat16)
        x_mlu = copy.deepcopy(x)

        out_cpu = torch.repeat_interleave(x, 2)
        out_mlu = torch.repeat_interleave(x_mlu.to("mlu"), 2)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-4, use_MSE=True
        )

        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        out_cpu.backward(grad)
        out_mlu.backward(grad.to("mlu"))
        self.assertTensorsEqual(x.grad, x_mlu.grad.cpu(), 3e-4, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
