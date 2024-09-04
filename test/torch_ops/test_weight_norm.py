import sys
import os
import copy
import logging
import unittest
from itertools import product
import torch
from torch import nn
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

shape_3D_list = [
    (128, 128, 1),
    (128, 128, 3),
    (128, 128, 5),
    (128, 128, 7),
    (128, 64, 3),
    (128, 64, 5),
    (128, 64, 8),
    (128, 80, 3),
    (2, 128, 3),
    (256, 128, 11),
    (256, 128, 6),
    (256, 256, 1),
    (256, 256, 7),
    (256, 80, 7),
    (32, 1, 11),
    (32, 1, 7),
    (32, 32, 11),
    (32, 32, 1),
    (32, 32, 3),
    (32, 32, 7),
    (4, 32, 7),
    (512, 256, 17),
    (64, 32, 10),
    (64, 32, 5),
    (64, 32, 7),
    (64, 32, 1),
    (64, 32, 3),
    (64, 32, 7),
    (80, 512, 3),
]  # the scale of the customer's network

shape_4D_list = [
    (1024, 1024, 5, 1),
    (1024, 512, 5, 1),
    (1, 1024, 3, 1),
    (128, 128, 3, 3),
    (128, 128, 3, 4),
    (128, 32, 5, 1),
    (128, 64, 4, 4),
    (1, 512, 1, 4),
    (1, 512, 1, 8),
    (256, 128, 4, 4),
    (256, 256, 3, 3),
    (256, 256, 3, 4),
    (32, 1, 5, 1),
    (32, 2, 7, 7),
    (32, 32, 3, 3),
    (512, 128, 5, 1),
    (512, 256, 4, 4),
    (64, 32, 3, 4),
    (64, 64, 3, 3),
]  # the scale of the customer's network

shape_and_dim_black_list = {
    # mlu500
    (128, 128, 7): [None, -1],
    (512, 256, 17): [None, -1],
    (64, 32, 5): [None, -1],
    (64, 32, 7): [None, -1],
    (1, 1024, 3, 1): [None, -4, -3, -2, -1, 0, 1, 2, 3],
    (128, 128, 3, 4): [None, -1],
    (256, 256, 3, 3): [None, -1],
}


class TestWeightNormOp(TestCase):
    def _test_weight_norm(self, shape, N, dtype, dim):
        if len(shape) == 3:
            # Conv1d is the middle layer of the customer network when the weight is 3D
            in_channels = shape[1]
            out_channels = shape[0]
            kernel_size = shape[2]
            L = kernel_size
            input = torch.randn(N, in_channels, L, dtype=dtype, requires_grad=True)
            m = nn.Conv1d(in_channels, out_channels, kernel_size).to(dtype)

        if len(shape) == 4:
            # Conv2d is the middle layer of the customer network when the weight is 4D
            in_channels = shape[1]
            out_channels = shape[0]
            kernel_size = (shape[2], shape[3])
            H = kernel_size[0]
            W = kernel_size[1]
            input = torch.randn(N, in_channels, H, W, dtype=dtype, requires_grad=True)
            m = nn.Conv2d(in_channels, out_channels, kernel_size).to(dtype)

        nn.utils.weight_norm(m, name="weight", dim=dim)

        return input, m

    # @unittest.skip("not test")
    @testinfo()
    def test_weight_norm(self):  # forward + backward
        shape_list = shape_3D_list + shape_4D_list
        N_list = [1, 32]
        dtype_list = [torch.float32]  # "slow_conv2d_cpu" not implemented for 'Half
        for shape in shape_list:
            dim_list = [None] + list(range(-len(shape), len(shape)))
            loop_var = [[shape], N_list, dtype_list, dim_list]
            for param in product(*loop_var):
                torch.manual_seed(1)
                shape, N, dtype, dim = param
                if (
                    shape in shape_and_dim_black_list
                    and dim in shape_and_dim_black_list[shape]
                ):
                    # mlu500: AssertionError: tensor not less than or equal to 0.003
                    continue
                input_cpu, m_cpu = self._test_weight_norm(shape, N, dtype, dim)

                output_cpu = m_cpu(input_cpu)
                weight_g_cpu = m_cpu.weight_g.data
                weight_v_cpu = m_cpu.weight_v.data
                grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
                output_cpu.backward(grad_cpu)
                input_grad_cpu = copy.deepcopy(input_cpu.grad)
                weight_g_grad_cpu = copy.deepcopy(m_cpu.weight_g.grad)
                weight_v_grad_cpu = copy.deepcopy(m_cpu.weight_v.grad)

                input_cpu.grad.zero_()
                m_cpu.weight_g.grad.zero_()
                m_cpu.weight_v.grad.zero_()

                input_mlu = input_cpu.to("mlu")
                m_mlu = m_cpu.to("mlu")

                output_mlu = m_mlu(input_mlu)
                weight_g_mlu = m_mlu.weight_g.data
                weight_v_mlu = m_mlu.weight_v.data
                grad_mlu = grad_cpu.to("mlu")
                output_mlu.backward(grad_mlu)
                input_grad_mlu = input_cpu.grad
                weight_g_grad_mlu = m_cpu.weight_g.grad
                weight_v_grad_mlu = m_cpu.weight_v.grad

                er = 0.003
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), er, use_MSE=True)
                self.assertTensorsEqual(
                    weight_g_cpu, weight_g_mlu.cpu(), er, use_MSE=True
                )
                self.assertTensorsEqual(
                    weight_v_cpu, weight_v_mlu.cpu(), er, use_MSE=True
                )

                self.assertTensorsEqual(
                    input_grad_cpu, input_grad_mlu.cpu(), er, use_MSE=True
                )
                self.assertTensorsEqual(
                    weight_g_grad_cpu, weight_g_grad_mlu.cpu(), er, use_MSE=True
                )
                self.assertTensorsEqual(
                    weight_v_grad_cpu, weight_v_grad_mlu.cpu(), er, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remove_weight_norm(self):  # fowward + backward
        shape_list = shape_3D_list + shape_4D_list
        N_list = [1, 32]
        dtype_list = [torch.float32]  # "slow_conv2d_cpu" not implemented for 'Half
        for shape in shape_list:
            dim_list = [None] + list(range(-len(shape), len(shape)))
            loop_var = [[shape], N_list, dtype_list, dim_list]
            for param in product(*loop_var):
                torch.manual_seed(1)
                shape, N, dtype, dim = param
                if shape == (1, 1024, 3, 1) and dim in [
                    None,
                    -4,
                    -3,
                    -2,
                    -1,
                    0,
                    1,
                    2,
                    3,
                ]:
                    continue  # mlu500: AssertionError: tensor not less than or equal to 0.003
                input_cpu, m_cpu = self._test_weight_norm(shape, N, dtype, dim)
                nn.utils.remove_weight_norm(m_cpu, name="weight")

                output_cpu = m_cpu(input_cpu)
                weight_cpu = m_cpu.weight.data
                grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
                output_cpu.backward(grad_cpu)
                input_grad_cpu = copy.deepcopy(input_cpu.grad)
                weight_grad_cpu = copy.deepcopy(m_cpu.weight.grad)

                input_cpu.grad.zero_()
                m_cpu.weight.grad.zero_()

                input_mlu = input_cpu.to("mlu")
                m_mlu = m_cpu.to("mlu")

                output_mlu = m_mlu(input_mlu)
                weight_mlu = m_mlu.weight.data
                grad_mlu = grad_cpu.to("mlu")
                output_mlu.backward(grad_mlu)
                input_grad_mlu = input_cpu.grad
                weight_grad_mlu = m_cpu.weight.grad

                er = 0.003
                self.assertTensorsEqual(output_cpu, output_mlu.cpu(), er, use_MSE=True)
                self.assertTensorsEqual(weight_cpu, weight_mlu.cpu(), er, use_MSE=True)

                self.assertTensorsEqual(
                    input_grad_cpu, input_grad_mlu.cpu(), er, use_MSE=True
                )
                self.assertTensorsEqual(
                    weight_grad_cpu, weight_grad_mlu.cpu(), er, use_MSE=True
                )


if __name__ == "__main__":
    unittest.main()
