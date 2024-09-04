from __future__ import print_function

import sys
import os
import logging
import copy
import unittest
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

type_list = [torch.float, torch.half]


class TestHardswishOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_contiguous(self):
        shape_list = [(168), (45, 50), (15, 224, 224), (1, 3, 224, 224)]
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                hardswish_layer = nn.Hardswish()
                output_cpu = hardswish_layer(data)
                output_mlu = hardswish_layer(self.to_mlu(x))
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                # test function:
                output_cpu_f = F.hardswish(data, inplace=False)
                output_mlu_f = F.hardswish(self.to_mlu(x), inplace=False)
                self.assertTensorsEqual(
                    output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_permute(self):
        import random

        for in_shape in [
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16, 4),
            (1, 3, 16, 16, 3, 6),
            (1, 3, 16, 16, 4, 15, 8),
        ]:
            input_ = torch.randn(in_shape, dtype=torch.float)
            size = np.arange(len(in_shape))
            random.shuffle(size)
            input_mlu = input_.to("mlu")
            input_ = torch.permute(input_, tuple(size))
            input_mlu = torch.permute(input_mlu, tuple(size))
            input_inplace_ = copy.deepcopy(input_)
            input_mlu_inplace_ = copy.deepcopy(input_mlu)
            output_cpu = F.hardswish(input_, inplace=False)
            output_mlu = F.hardswish(input_mlu, inplace=False)
            F.hardswish(input_inplace_, inplace=True)  # test inplace operation
            F.hardswish(input_mlu_inplace_, inplace=True)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_inplace_, input_mlu_inplace_.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_channel_last(self):
        shape_list = [(1, 3, 224, 224), (4, 7, 15, 224, 224)]
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_mlu = x.mlu()
                x_mlu = self.convert_to_channel_last(x_mlu)
                hardswish_layer = nn.Hardswish()
                output_cpu = hardswish_layer(data)
                output_mlu = hardswish_layer(x_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                # test function:
                output_cpu_f = F.hardswish(data, inplace=False)
                output_mlu_f = F.hardswish(x_mlu, inplace=False)
                self.assertTensorsEqual(
                    output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_not_dense(self):
        shape_list_not_dense = [(45, 100), (15, 224, 448), (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x_cpu = copy.deepcopy(data)[..., 0 : in_shape[-1] // 2]
                x = data.to(typeId)
                x_mlu = copy.deepcopy(x).mlu()
                x_mlu = x_mlu[..., 0 : in_shape[-1] // 2]
                hardswish_layer = nn.Hardswish()
                output_cpu = hardswish_layer(x_cpu)
                output_mlu = hardswish_layer(x_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                # test function:
                output_cpu_f = F.hardswish(x_cpu, inplace=False)
                output_mlu_f = F.hardswish(x_mlu, inplace=False)
                self.assertTensorsEqual(
                    output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_inplace_contiguous(self):
        shape_list = [(168), (45, 50), (15, 224, 224), (1, 3, 224, 224)]
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_cpu = copy.deepcopy(data)
                x_mlu = copy.deepcopy(x)
                x_mlu = self.to_mlu(x_mlu)
                x_mlu_data_ptr = x_mlu.data_ptr()
                F.hardswish(x_mlu, inplace=True)
                F.hardswish(x_cpu, inplace=True)
                self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_inplace_channel_last(self):
        shape_list = [(1, 3, 224, 224), (4, 7, 15, 224, 224)]
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_cpu = copy.deepcopy(data)
                x_mlu = x.mlu()
                x_mlu = self.convert_to_channel_last(x_mlu)
                x_mlu_data_ptr = x_mlu.data_ptr()
                F.hardswish(x_mlu, inplace=True)
                F.hardswish(x_cpu, inplace=True)
                self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_inplace_not_dense(self):
        shape_list_not_dense = [(45, 100), (15, 224, 448), (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_cpu = copy.deepcopy(data)[..., 0 : in_shape[-1] // 2]
                x_mlu = copy.deepcopy(x).mlu()
                x_mlu = x_mlu[..., 0 : in_shape[-1] // 2]
                x_mlu_data_ptr = x_mlu.data_ptr()
                F.hardswish(x_mlu, inplace=True)
                F.hardswish(x_cpu, inplace=True)
                self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_backward(self):
        shape_list = [(168), (45, 50), (15, 224, 224), (1, 3, 224, 224)]
        for shape in shape_list:
            data = torch.randn(shape, dtype=torch.float, requires_grad=True)
            x_mlu = data.mlu()
            hardswish_layer = nn.Hardswish()
            out_cpu = hardswish_layer(data)
            out_mlu = hardswish_layer(x_mlu)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu = grad.mlu()

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(data.grad)
            data.grad.zero_()

            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(data.grad)

            self.assertTensorsEqual(
                out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_backwark_permute(self):
        input_cpu = torch.randn((4, 3, 2, 1), dtype=torch.float, requires_grad=True)
        out_cpu = F.hardswish(input_cpu)
        out_mlu = F.hardswish(input_cpu.to("mlu"))
        grad = torch.randn((3, 2, 1, 4), dtype=torch.float)  # test backward
        out_cpu.backward(torch.permute(grad, (3, 0, 1, 2)))
        grad_cpu = copy.deepcopy(input_cpu.grad)
        input_cpu.grad.zero_()
        grad_mlu = grad.mlu().permute(3, 0, 1, 2)
        out_mlu.backward(grad_mlu)
        grad_mlu = input_cpu.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardswish_special(self):
        shape_list = [(), (1,), (15,), (2, 3), (1, 1, 1), (1, 1, 1, 1)]
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                hardswish_layer = nn.Hardswish()
                output_cpu = hardswish_layer(data)
                output_mlu = hardswish_layer(self.to_mlu(x))
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003)
                # test function:
                output_cpu_f = F.hardswish(data, inplace=False)
                output_mlu_f = F.hardswish(self.to_mlu(x), inplace=False)
                self.assertTensorsEqual(output_cpu_f, output_mlu_f.cpu(), 0.003)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_hardswish_bfloat16(self):
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        data = torch.tensor(inputValues, dtype=torch.bfloat16, requires_grad=True)
        x_mlu = data.mlu()
        hardswish_layer = nn.Hardswish()
        out_cpu = hardswish_layer(data)
        out_mlu = hardswish_layer(x_mlu)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        grad_mlu = grad.mlu()
        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(data.grad)
        data.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = copy.deepcopy(data.grad)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            out_grad_cpu.float(), out_grad_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
