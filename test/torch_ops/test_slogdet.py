from __future__ import print_function
import sys
import os

import unittest
import copy

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
import logging  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestLogdetOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_slogdet_out(self):
        shape_list = [
            [1, 4, 4],
            [5, 3, 5, 5],
            [5, 0, 4, 4],
            [0, 3, 3],
            [0, 0],
            [5, 5, 0, 0],
            [5, 3, 5, 5],
        ]
        type_list = [torch.float32, torch.double]
        func_list = [torch.slogdet, torch.linalg.slogdet]
        for func in func_list:
            for type in type_list:
                for shape in shape_list:
                    x = torch.rand(shape, dtype=type)
                    x_copy = copy.deepcopy(x)
                    x_mlu = x_copy.to("mlu")

                    out_mlu_logdet = torch.zeros((1), dtype=type).mlu()
                    out_mlu_sign = torch.zeros((1), dtype=type).mlu()

                    out_cpu_logdet = torch.empty((1), dtype=type)
                    out_cpu_sign = torch.empty((1), dtype=type)
                    func(x, out=[out_cpu_sign, out_cpu_logdet])

                    func(x_mlu, out=[out_mlu_sign, out_mlu_logdet])

                    for i in range(out_cpu_sign.numel()):
                        cpu_res = out_cpu_sign.view(-1)
                        mlu_res = out_mlu_sign.cpu().view(-1)
                        if torch.isnan(cpu_res[i]):
                            continue
                        self.assertTensorsEqual(
                            cpu_res[i], mlu_res[i], 3e-3, use_MSE=True
                        )

                    for i in range(out_cpu_logdet.numel()):
                        cpu_res = out_cpu_logdet.view(-1)
                        mlu_res = out_mlu_logdet.cpu().view(-1)
                        if torch.isnan(cpu_res[i]):
                            continue
                        self.assertTensorsEqual(
                            cpu_res[i], mlu_res[i], 3e-3, use_MSE=True
                        )

    @testinfo()
    def test_slogdet(self):
        shape_list = [
            [1, 4, 4],
            [5, 3, 5, 5],
            [5, 0, 4, 4],
            [0, 3, 3],
            [0, 0],
            [5, 5, 0, 0],
        ]
        type_list = [torch.float32]
        func_list = [torch.slogdet, torch.linalg.slogdet]
        for func in func_list:
            for type in type_list:
                for shape in shape_list:
                    x = torch.rand(shape, dtype=type, requires_grad=True)
                    x_copy = copy.deepcopy(x)
                    x_mlu = x_copy.to("mlu")

                    out_cpu = func(x)
                    out_mlu = func(x_mlu)

                    for k in range(2):
                        for i in range(out_cpu[k].numel()):
                            cpu_res = out_cpu[k].view(-1)
                            mlu_res = out_mlu[k].view(-1)
                            if torch.isnan(cpu_res[i]):
                                continue
                            self.assertTensorsEqual(
                                cpu_res[i], mlu_res[i].cpu(), 3e-3, use_MSE=True
                            )

                    out_cpu.logabsdet.backward(out_cpu.logabsdet)
                    out_mlu.logabsdet.backward(out_mlu.logabsdet)

                    out_cpu = x.grad
                    out_mlu = x_copy.grad
                    for i in range(out_cpu.numel()):
                        cpu_res = out_cpu.view(-1)
                        mlu_res = out_mlu.cpu().view(-1)
                        if torch.isnan(cpu_res[i]):
                            continue
                        self.assertTensorsEqual(
                            cpu_res[i], mlu_res[i].cpu(), 3e-3, use_MSE=True
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_logdet_channels_last(self):
        shape_list = [(5, 3, 5, 5)]
        type_list = [torch.float32, torch.double]
        func_list = [torch.slogdet, torch.linalg.slogdet]
        for func in func_list:
            for type in type_list:
                for shape in shape_list:
                    x = torch.rand(shape, dtype=type)
                    x_copy = copy.deepcopy(x)
                    x_mlu = x_copy.to("mlu")
                    x = x.to(memory_format=torch.channels_last)
                    x_mlu = x_mlu.to(memory_format=torch.channels_last)
                    x.requires_grad = True
                    x_mlu.requires_grad = True
                    out_cpu = func(x)
                    out_mlu = func(x_mlu)
                    out_cpu.logabsdet.backward(out_cpu.logabsdet)
                    out_mlu.logabsdet.backward(out_mlu.logabsdet)

                    for k in range(2):
                        for i in range(out_cpu[k].numel()):
                            cpu_res = out_cpu[k].contiguous().view(-1)
                            mlu_res = out_mlu[k].cpu().contiguous().view(-1)
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


if __name__ == "__main__":
    unittest.main()
