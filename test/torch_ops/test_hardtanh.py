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
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)

shape_list = [
    (),
    (2),
    (15),
    (1, 1),
    (45, 50),
    (1, 1, 1),
    (15, 224, 224),
    (1, 1, 1, 1),
    (1, 3, 224, 224),
]
minmax_list = [(-0.2, 0.4), (-2, 2), (12, 24), (-24, -12)]
type_list = [torch.float, torch.half]


class TestHardtanhOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_contiguous(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)
                    output_cpu = hardtanh_layer(data)
                    output_mlu = hardtanh_layer(self.to_mlu(x))
                    self.assertTensorsEqual(
                        output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                    )
                    # test function:
                    output_cpu_f = F.hardtanh(data, min_v, max_v, inplace=False)
                    output_mlu_f = F.hardtanh(
                        self.to_mlu(x), min_v, max_v, inplace=False
                    )
                    self.assertTensorsEqual(
                        output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_permute(self):
        import random

        for in_shape in [
            (8, 224, 224),
            (1, 1, 1, 1),
            (1, 3, 16, 16, 4),
            (1, 3, 16, 16, 3, 6),
            (1, 3, 16, 16, 4, 15, 8),
        ]:
            for min_v, max_v in minmax_list:
                input_ = torch.randn(in_shape, dtype=torch.float)
                hardtanh_layer = nn.Hardtanh(min_v, max_v)
                size = np.arange(len(in_shape))
                random.shuffle(size)
                input_mlu = input_.to("mlu")
                input_ = torch.permute(input_, tuple(size))
                input_mlu = torch.permute(input_mlu, tuple(size))
                input_inplace_ = copy.deepcopy(input_)
                input_mlu_inplace_ = copy.deepcopy(input_mlu)
                output_cpu = F.hardtanh(input_, min_v, max_v, inplace=False)
                output_mlu = F.hardtanh(input_mlu, min_v, max_v, inplace=False)
                F.hardtanh(
                    input_inplace_, min_v, max_v, inplace=True
                )  # test inplace operation
                F.hardtanh(input_mlu_inplace_, min_v, max_v, inplace=True)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    input_inplace_, input_mlu_inplace_.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_channel_last(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x = self.convert_to_channel_last(x)
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)
                    output_cpu = hardtanh_layer(data)
                    output_mlu = hardtanh_layer(self.to_mlu(x))
                    self.assertTensorsEqual(
                        output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                    )
                    # test function:
                    output_cpu_f = F.hardtanh(data, min_v, max_v, inplace=False)
                    output_mlu_f = F.hardtanh(
                        self.to_mlu(x), min_v, max_v, inplace=False
                    )
                    self.assertTensorsEqual(
                        output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_not_dense(self):
        shape_list_not_dense = [(45, 100), (15, 224, 448), (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x_cpu = copy.deepcopy(data)[..., 0 : in_shape[-1] // 2]
                    x_mlu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x_mlu)[..., 0 : in_shape[-1] // 2]
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)
                    output_cpu = hardtanh_layer(x_cpu)
                    output_mlu = hardtanh_layer(x_mlu)
                    self.assertTensorsEqual(
                        output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                    )
                    # test function:
                    output_cpu_f = F.hardtanh(x_cpu, min_v, max_v, inplace=False)
                    output_mlu_f = F.hardtanh(x_mlu, min_v, max_v, inplace=False)
                    self.assertTensorsEqual(
                        output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_inplace_contiguous(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x_cpu = copy.deepcopy(data)
                    x_mlu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x_mlu)
                    x_mlu_data_ptr = x_mlu.data_ptr()
                    F.hardtanh(x_mlu, min_v, max_v, inplace=True)
                    F.hardtanh(x_cpu, min_v, max_v, inplace=True)
                    self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

                    x_mlu_2 = copy.deepcopy(x)
                    x_mlu_2 = self.to_mlu(x_mlu_2)
                    x_mlu_2_data_ptr = x_mlu_2.data_ptr()
                    F.hardtanh_(x_mlu_2, min_v, max_v)
                    self.assertEqual(x_mlu_2_data_ptr, x_mlu_2.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu_2.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_inplace_channel_last(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x = self.convert_to_channel_last(x)
                    x_cpu = copy.deepcopy(data)
                    x_mlu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x_mlu)
                    x_mlu_data_ptr = x_mlu.data_ptr()
                    F.hardtanh(x_mlu, min_v, max_v, inplace=True)
                    F.hardtanh(x_cpu, min_v, max_v, inplace=True)
                    self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

                    x_mlu_2 = copy.deepcopy(x)
                    x_mlu_2 = self.to_mlu(x_mlu_2)
                    x_mlu_2_data_ptr = x_mlu_2.data_ptr()
                    F.hardtanh_(x_mlu_2, min_v, max_v)
                    self.assertEqual(x_mlu_2_data_ptr, x_mlu_2.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu_2.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_inplace_not_dense(self):
        shape_list_not_dense = [(45, 100), (15, 224, 448), (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for min_v, max_v in minmax_list:
                for typeId in type_list:
                    data = torch.randn(in_shape, dtype=torch.float)
                    x = data.to(typeId)
                    x_cpu = copy.deepcopy(data)[..., 0 : in_shape[-1] // 2]
                    x_mlu = copy.deepcopy(x)
                    x_mlu = self.to_mlu(x_mlu)[..., 0 : in_shape[-1] // 2]
                    x_mlu_data_ptr = x_mlu.data_ptr()
                    F.hardtanh(x_mlu, min_v, max_v, inplace=True)
                    F.hardtanh(x_cpu, min_v, max_v, inplace=True)
                    self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

                    x_mlu_2 = copy.deepcopy(x)
                    x_mlu_2 = self.to_mlu(x_mlu_2)[..., 0 : in_shape[-1] // 2]
                    x_mlu_2_data_ptr = x_mlu_2.data_ptr()
                    F.hardtanh_(x_mlu_2, min_v, max_v)
                    self.assertEqual(x_mlu_2_data_ptr, x_mlu_2.data_ptr())
                    self.assertTensorsEqual(x_cpu, x_mlu_2.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_backward(self):
        for shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in [torch.float]:
                    # for typeId in type_list:
                    data = torch.randn(shape, dtype=torch.float, requires_grad=True)
                    x = data.to(typeId)
                    x_mlu = x.to("mlu")
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)

                    out_cpu = hardtanh_layer(data)
                    out_mlu = hardtanh_layer(x_mlu)
                    grad = torch.randn(out_cpu.shape, dtype=torch.float)
                    grad_mlu = grad.to("mlu")

                    out_cpu.backward(grad)
                    out_grad_cpu = copy.deepcopy(data.grad)
                    data.grad.zero_()
                    out_mlu.backward(grad_mlu)
                    out_grad_mlu = copy.deepcopy(data.grad)
                    if typeId == torch.float16:
                        self.assertTensorsEqual(
                            out_grad_cpu, out_grad_mlu.cpu().float(), 0.02, use_MSE=True
                        )
                    else:
                        self.assertTensorsEqual(
                            out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardtanh_backwark_permute(self):
        input_cpu = torch.randn((4, 3, 2, 1), dtype=torch.float, requires_grad=True)
        out_cpu = F.hardtanh(input_cpu, 0.1, 2)
        out_mlu = F.hardtanh(input_cpu.to("mlu"), 0.1, 2)
        grad = torch.randn((3, 2, 1, 4), dtype=torch.float)  # test backward
        out_cpu.backward(torch.permute(grad, (3, 0, 1, 2)))
        grad_cpu = copy.deepcopy(input_cpu.grad)
        input_cpu.grad.zero_()

        out_mlu.backward(grad.to("mlu").permute(3, 0, 1, 2))
        grad_mlu = input_cpu.grad
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("86GB")
    def test_hardtanh_backward_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        minmax_list = [(-2, 2)]
        for shape in shape_list:
            for min_v, max_v in minmax_list:
                for typeId in [torch.float]:
                    # for typeId in type_list:
                    data = torch.randn(shape, dtype=torch.float, requires_grad=True)
                    x = data.to(typeId)
                    x_mlu = x.to("mlu")
                    hardtanh_layer = nn.Hardtanh(min_v, max_v)

                    out_cpu = hardtanh_layer(data)
                    out_mlu = hardtanh_layer(x_mlu)
                    grad = torch.randn(out_cpu.shape, dtype=torch.float)
                    grad_mlu = grad.to("mlu")

                    out_cpu.backward(grad)
                    out_grad_cpu = copy.deepcopy(data.grad)
                    data.grad.zero_()
                    out_mlu.backward(grad_mlu)
                    out_grad_mlu = copy.deepcopy(data.grad)
                    self.assertTensorsEqual(
                        out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_hardtanh_bfloat16(self):
        for in_shape in shape_list:
            for min_v, max_v in minmax_list:
                data = torch.randn(in_shape, dtype=torch.bfloat16, requires_grad=True)
                # x = data.to(typeId)
                hardtanh_layer = nn.Hardtanh(min_v, max_v)
                output_cpu = hardtanh_layer(data)
                output_mlu = hardtanh_layer(self.to_mlu(data))

                grad_cpu = torch.randn(output_cpu.shape, dtype=torch.bfloat16)
                grad_mlu = grad_cpu.to("mlu")
                output_cpu.backward(grad_cpu)
                out_grad_cpu = copy.deepcopy(data.grad)
                data.grad.zero_()
                output_mlu.backward(grad_mlu)
                out_grad_mlu = copy.deepcopy(data.grad)

                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_grad_cpu, out_grad_mlu.cpu(), 0.003, use_MSE=True
                )

                # test function:
                output_cpu_f = F.hardtanh(data, min_v, max_v, inplace=False)
                output_mlu_f = F.hardtanh(
                    self.to_mlu(data), min_v, max_v, inplace=False
                )

                grad_cpu_f = torch.randn(output_cpu_f.shape, dtype=torch.bfloat16)
                grad_mlu_f = grad_cpu_f.to("mlu")
                data.grad.zero_()
                output_cpu_f.backward(grad_cpu_f)
                out_grad_cpu_f = copy.deepcopy(data.grad)
                data.grad.zero_()
                output_mlu_f.backward(grad_mlu_f)
                out_grad_mlu_f = copy.deepcopy(data.grad)

                self.assertTensorsEqual(
                    output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_grad_cpu_f, out_grad_mlu_f.cpu(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
