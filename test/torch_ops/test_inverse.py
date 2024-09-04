import sys
import os
import math
import unittest
import logging
import copy

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(1234)


class TestInvOp(TestCase):
    @testinfo()
    def test_inv(self):
        shape_list = [(1, 4, 4), (3, 3), (4, 5, 3, 3), (0, 3, 3), (0, 0)]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.rand(shape, dtype=type, requires_grad=True)
                x_copy = copy.deepcopy(x)
                x_mlu = x_copy.to("mlu")
                out_cpu = torch.inverse(x)
                out_mlu = torch.inverse(x_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)
                out_cpu.backward(out_cpu)
                out_mlu.backward(out_mlu)
                self.assertTensorsEqual(x.grad, x_copy.grad, 3e-3, use_MSE=True)

    @testinfo()
    def test_inv_channelslast(self):
        shape_list = [(5, 3, 3, 3)]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.rand(shape, dtype=type, requires_grad=True)
                x_copy = copy.deepcopy(x)
                x_mlu = x_copy.to("mlu")
                x_cl = x.to(memory_format=torch.channels_last)
                x_mlu_cl = x_mlu.to(memory_format=torch.channels_last)
                out_cpu = torch.inverse(x_cl.permute([0, 2, 3, 1]))
                out_mlu = torch.inverse(x_mlu_cl.permute([0, 2, 3, 1]))
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    def test_inv_empty_input(self):
        x = torch.randn(0, 0).cpu()
        x_mlu = x.mlu()
        out_cpu = torch.inverse(x)
        out_mlu = torch.inverse(x_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.00, use_MSE=False)

    @testinfo()
    def test_inv_not_dense(self):
        input = torch.rand(4, 4).to(torch.float)
        value = math.nan
        x_cpu = input.new_empty(input.shape + (2,))
        x_cpu[..., 0] = value
        x_cpu[..., 1] = input.detach()
        x_cpu = x_cpu[..., 1]
        input_mlu = input.mlu()
        x_mlu = input_mlu.new_empty(input_mlu.shape + (2,))
        x_mlu[..., 0] = value
        x_mlu[..., 1] = input_mlu.detach()
        x_mlu = x_mlu[..., 1]
        out_cpu = torch.inverse(x_cpu)
        out_mlu = torch.inverse(x_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    @testinfo()
    def test_inv_out(self):
        x = torch.rand(4, 4).to(torch.float)
        out_cpu = torch.zeros(1, 16).to(torch.float)
        x_mlu = x.mlu()
        out_mlu = torch.zeros(1, 16).to(torch.float).mlu()
        torch.inverse(x, out=out_cpu)
        torch.inverse(x_mlu, out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
