from __future__ import print_function

import sys
import os
import logging
import copy
import unittest
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

shape_list = [
    (150),
    (10, 90),
    (45, 50),
    (10, 20, 32),
    (15, 224, 224),
    (2, 32, 128, 128),
    (1, 3, 224, 224),
]
type_list = [torch.float, torch.half]


class TestHardsigmoidOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_contiguous(self):
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                hardsigmoid_layer = nn.Hardsigmoid()
                output_cpu = hardsigmoid_layer(data)
                output_mlu = hardsigmoid_layer(self.to_mlu(x))
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                # test function:
                output_cpu_f = F.hardsigmoid(data, inplace=False)
                output_mlu_f = F.hardsigmoid(self.to_mlu(x), inplace=False)
                self.assertTensorsEqual(
                    output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_permute(self):
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
            output_cpu = F.hardsigmoid(input_, inplace=False)
            output_mlu = F.hardsigmoid(input_mlu, inplace=False)
            F.hardsigmoid(input_inplace_, inplace=True)  # test inplace operation
            F.hardsigmoid(input_mlu_inplace_, inplace=True)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input_inplace_, input_mlu_inplace_.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_channel_last(self):
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_mlu = x.mlu()
                x_mlu = self.convert_to_channel_last(x_mlu)
                hardsigmoid_layer = nn.Hardsigmoid()
                output_cpu = hardsigmoid_layer(data)
                output_mlu = hardsigmoid_layer(x_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                # test function:
                output_cpu_f = F.hardsigmoid(data, inplace=False)
                output_mlu_f = F.hardsigmoid(x_mlu, inplace=False)
                self.assertTensorsEqual(
                    output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_not_dense(self):
        shape_list_not_dense = [(45, 100), (15, 224, 448), (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x_cpu = copy.deepcopy(data)[..., 0 : in_shape[-1] // 2]
                x = data.to(typeId)
                x_mlu = copy.deepcopy(x).mlu()
                x_mlu = x_mlu[..., 0 : in_shape[-1] // 2]
                hardsigmoid_layer = nn.Hardsigmoid()
                output_cpu = hardsigmoid_layer(x_cpu)
                output_mlu = hardsigmoid_layer(x_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 0.003, use_MSE=True
                )
                # test function:
                output_cpu_f = F.hardsigmoid(x_cpu, inplace=False)
                output_mlu_f = F.hardsigmoid(x_mlu, inplace=False)
                self.assertTensorsEqual(
                    output_cpu_f, output_mlu_f.cpu(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_inplace_contiguous(self):
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_cpu = copy.deepcopy(data)
                x_mlu = copy.deepcopy(x)
                x_mlu = self.to_mlu(x_mlu)
                x_mlu_data_ptr = x_mlu.data_ptr()
                F.hardsigmoid(x_mlu, inplace=True)
                F.hardsigmoid(x_cpu, inplace=True)
                self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_inplace_channel_last(self):
        for in_shape in shape_list:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_cpu = copy.deepcopy(data)
                x_mlu = x.mlu()
                x_mlu = self.convert_to_channel_last(x_mlu)
                x_mlu_data_ptr = x_mlu.data_ptr()
                F.hardsigmoid(x_mlu, inplace=True)
                F.hardsigmoid(x_cpu, inplace=True)
                self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_inplace_not_dense(self):
        shape_list_not_dense = [(45, 100), (15, 224, 448), (1, 3, 224, 448)]
        for in_shape in shape_list_not_dense:
            for typeId in type_list:
                data = torch.randn(in_shape, dtype=torch.float)
                x = data.to(typeId)
                x_cpu = copy.deepcopy(data)[..., 0 : in_shape[-1] // 2]
                x_mlu = copy.deepcopy(x).mlu()
                x_mlu = x_mlu[..., 0 : in_shape[-1] // 2]
                x_mlu_data_ptr = x_mlu.data_ptr()
                F.hardsigmoid(x_mlu, inplace=True)
                F.hardsigmoid(x_cpu, inplace=True)
                self.assertEqual(x_mlu_data_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_hardsigmoid_backward(self):
        for shape in shape_list:
            data = torch.randn(shape, dtype=torch.float, requires_grad=True)
            x_mlu = data.mlu()
            hardsigmoid_layer = nn.Hardsigmoid()
            out_cpu = hardsigmoid_layer(data)
            out_mlu = hardsigmoid_layer(x_mlu)
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
    def test_hardsigmoid_backwark_permute(self):
        input_cpu = torch.randn((4, 3, 2, 1), dtype=torch.float, requires_grad=True)
        out_cpu = F.hardsigmoid(input_cpu)
        out_mlu = F.hardsigmoid(input_cpu.to("mlu"))
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
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_hardsigmoid_bfloat16(self):
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        data = torch.tensor(inputValues, dtype=torch.bfloat16)
        x_mlu = data.mlu()
        hardsigmoid_layer = nn.Hardsigmoid()
        out_cpu = hardsigmoid_layer(data)
        out_mlu = hardsigmoid_layer(x_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_hardsigmoid_backward_bfloat16(self):
        inputValues = [-3.0, 3.0, -2.0, 2.0, -6.0, 6.0]
        expectedValues = [0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0]
        inputTensor = torch.tensor(
            inputValues, dtype=torch.bfloat16, device="mlu"
        ).requires_grad_()
        expetedTensor = torch.tensor(expectedValues, dtype=torch.bfloat16, device="mlu")
        out = F.hardsigmoid(inputTensor)
        out.backward(torch.ones_like(inputTensor))
        # precision override: refer from native ci
        self.assertEqual(inputTensor.grad, expetedTensor, atol=1e-2, rtol=0)


if __name__ == "__main__":
    unittest.main()
