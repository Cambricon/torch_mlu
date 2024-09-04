from __future__ import print_function
import sys
import os

import unittest
import copy

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413, C0411

import logging  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestLogdetOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_logdet_basic(self):
        shape_list = [(1, 4, 4), (3, 3), (4, 5, 3, 3)]
        type_list = [torch.float32]
        for type in type_list:
            for shape in shape_list:
                x = torch.rand(shape, dtype=type, requires_grad=True)
                x_copy = copy.deepcopy(x)
                x_mlu = x_copy.to("mlu")
                out_cpu = torch.logdet(x)
                out_mlu = torch.logdet(x_mlu)
                out_cpu.backward(out_cpu)
                out_mlu.backward(out_mlu)

                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.view(-1)
                    mlu_res = out_mlu.cpu().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(cpu_res[i], mlu_res[i], 3e-3, use_MSE=True)
                out_cpu = x.grad
                out_mlu = x_copy.grad
                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.view(-1)
                    mlu_res = out_mlu.cpu().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(cpu_res[i], mlu_res[i], 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_logdet_channels_last(self):
        shape_list = [(5, 3, 5, 5)]
        type_list = [torch.float32, torch.double]
        for type in type_list:
            for shape in shape_list:
                x = torch.rand(shape, dtype=type)
                x_copy = copy.deepcopy(x)
                x_mlu = x_copy.to("mlu")
                x = x.to(memory_format=torch.channels_last)
                x_mlu = x_mlu.to(memory_format=torch.channels_last)
                x.requires_grad = True
                x_mlu.requires_grad = True
                out_cpu = torch.logdet(x)
                out_mlu = torch.logdet(x_mlu)
                out_cpu.backward(out_cpu)
                out_mlu.backward(out_mlu)

                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.contiguous().view(-1)
                    mlu_res = out_mlu.cpu().contiguous().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(
                        cpu_res[i], mlu_res.cpu()[i], 3e-3, use_MSE=True
                    )
                out_cpu = x.grad
                out_mlu = x_mlu.grad
                for i in range(out_cpu.numel()):
                    cpu_res = out_cpu.contiguous().view(-1)
                    mlu_res = out_mlu.cpu().contiguous().view(-1)
                    if torch.isnan(cpu_res[i]):
                        continue
                    self.assertTensorsEqual(
                        cpu_res[i], mlu_res.cpu()[i], 3e-3, use_MSE=True
                    )
        # test empty input
        x = torch.randn(0, 0)
        out_cpu = torch.logdet(x)
        out_mlu = torch.logdet(x.to("mlu"))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)


if __name__ == "__main__":
    unittest.main()
