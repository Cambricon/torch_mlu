# pylint: disable=C0413,C0411
from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase, read_card_info, skipBFloat16IfNotSupport
from itertools import product

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestEluOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_elu(self):
        shape_list = [
            (6),
            (24, 8),
            (4, 12, 38),
            (128, 4, 128, 124),
            (2, 5, 11, 14, 22),
            (1, 3, 6, 10, 20, 22),
        ]
        alpha_list = [0.5, 1, 1.5, 2]
        type_list = [torch.float, torch.half]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        list_list = [alpha_list, type_list, shape_list, func_list]
        m = torch.nn.ELU()
        m_place = torch.nn.ELU(inplace=True)
        for alpha, dtype, shape, func in product(*list_list):
            m_alpha = torch.nn.ELU(alpha=alpha)
            input_ = torch.randn(shape, dtype=dtype)
            input_cpu = func(copy.deepcopy(input_)).float()
            input_mlu = func(input_.to("mlu"))
            output_cpu = m(input_cpu)
            output_mlu = m(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, input_, 0, use_MSE=True)

            output_cpu = m_alpha(input_cpu)
            output_mlu = m_alpha(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)

            input_mlu_data_ptr = input_mlu.data_ptr()
            m_place(input_cpu)
            m_place(input_mlu)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003, use_MSE=True)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_elu_backward(self):
        in_shape = [
            (50),
            (35, 46),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (16, 17, 18, 19, 20, 21),
        ]
        alpha_list = [0.5, 1, 1.5, 2]
        list_list = [alpha_list, in_shape]
        for alpha, shape in product(*list_list):
            m = torch.nn.ELU(alpha=alpha)
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            x_mlu = x.to("mlu")

            out_cpu = m(x)
            out_mlu = m(x_mlu)

            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu = grad.to("mlu")

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()

            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu().float(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_celu(self):
        shape_list = [
            (6),
            (24, 8),
            (4, 12, 38),
            (128, 4, 128, 124),
            (2, 5, 11, 14, 22),
            (1, 3, 6, 10, 20, 22),
        ]
        alpha_list = [0.5, 1, 1.5, 2]
        type_list = [torch.float, torch.half]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        list_list = [alpha_list, type_list, shape_list, func_list]
        m = torch.nn.CELU()
        m_place = torch.nn.CELU(inplace=True)
        for alpha, dtype, shape, func in product(*list_list):
            m_alpha = torch.nn.CELU(alpha=alpha)
            input_ = torch.randn(shape, dtype=dtype)
            input_cpu = func(copy.deepcopy(input_)).float()
            input_mlu = func(input_.to("mlu"))
            output_cpu = m(input_cpu)
            output_mlu = m(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input_cpu, input_, 0, use_MSE=True)

            output_cpu = m_alpha(input_cpu)
            output_mlu = m_alpha(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)

            input_mlu_data_ptr = input_mlu.data_ptr()
            m_place(input_cpu)
            m_place(input_mlu)
            self.assertTensorsEqual(input_cpu, input_mlu.cpu(), 0.003, use_MSE=True)
            self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_celu_backward(self):
        in_shape = [
            (50),
            (35, 46),
            (16, 27, 38),
            (128, 4, 128, 124),
            (14, 19, 11, 13, 21),
            (6, 7, 8, 9, 10, 11),
            (16, 17, 18, 19, 20, 21),
        ]
        alpha_list = [0.5, 1, 1.5, 2]
        list_list = [alpha_list, in_shape]
        for alpha, shape in product(*list_list):
            m = torch.nn.CELU(alpha=alpha)
            x = torch.randn(shape, dtype=torch.float, requires_grad=True)
            x_mlu = x.to("mlu")

            out_cpu = m(x)
            out_mlu = m(x_mlu)

            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu = grad.to("mlu")

            out_cpu.backward(grad)
            out_grad_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()

            out_mlu.backward(grad_mlu)
            out_grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu().float(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_elu_boundary_value(self):
        for number in [0, 0.0001, -0.0001, 9999]:
            for dtype in [torch.float, torch.half]:
                x = torch.tensor(number, dtype=dtype)
                output_cpu = F.elu(x.float())
                input_cpu = copy.deepcopy(x).float()
                output_mlu = F.elu(x.to("mlu"))
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )

                input_mlu = x.to("mlu")  # test inplace operation
                input_mlu_data_ptr = input_mlu.data_ptr()
                F.elu(input_cpu, inplace=True)
                F.elu(input_mlu, inplace=True)
                self.assertTensorsEqual(
                    input_cpu, input_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertEqual(input_mlu_data_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_elu_bfloat16(self):
        shape = [1, 16, 27, 38]
        alpha = 0.5
        m = torch.nn.ELU(alpha=alpha)
        x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
        x_mlu = x.to("mlu")
        out_cpu = m(x)
        out_mlu = m(x_mlu)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        grad_mlu = grad.to("mlu")
        out_cpu.backward(grad)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu.backward(grad_mlu)
        out_grad_mlu = copy.deepcopy(x.grad)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003)
        self.assertTensorsEqual(out_grad_cpu, out_grad_mlu.cpu(), 0.003)


if __name__ == "__main__":
    unittest.main()
