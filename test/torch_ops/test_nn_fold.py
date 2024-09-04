import sys
import os
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestFoldOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_nn_fold(self):
        # only support 3 dims or 4 dims input.
        N_lst = [0, 1, 5]
        Ci_lst = [1, 3]
        H_lst = [20, 32]
        W_lst = [22, 34]
        K_lst = [1, 3]
        padding_lst = [1, 2]
        stride_lst = [1, 4]
        dilation_lst = [1, 3]
        dtype_list = [torch.float, torch.half]
        func_lst = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        product_list = product(
            N_lst,
            Ci_lst,
            H_lst,
            W_lst,
            K_lst,
            padding_lst,
            stride_lst,
            dilation_lst,
            dtype_list,
            func_lst,
        )
        for N, Ci, H, W, K, P, S, D, dtype, func in product_list:
            H_O = int(((H + 2 * P) - (D * (K - 1) + 1)) / S + 1)
            W_O = int(((W + 2 * P) - (D * (K - 1) + 1)) / S + 1)
            input_cpu, input_mlu = self.TensorGenerator(
                (N, Ci * K * K, H_O * W_O), dtype, func
            )
            input_cpu = torch.nn.Parameter(input_cpu)
            input_mlu = torch.nn.Parameter(input_mlu)
            fold = torch.nn.Fold(
                output_size=(H, W),
                kernel_size=(K, K),
                stride=(S, S),
                padding=(P, P),
                dilation=(D, D),
            )
            output_cpu = fold(input_cpu)
            output_mlu = fold(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            grad_cpu, grad_mlu = self.TensorGenerator(output_cpu.size(), dtype, func)
            output_cpu.backward(grad_cpu)
            output_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                input_cpu.grad, input_mlu.grad.cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_fold_invalid_arg(self):
        # input wrong dimension

        fold = torch.nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
        with self.assertRaisesRegex(
            RuntimeError, r"be divisible by the product of kernel_size"
        ):
            fold(torch.randn(1, 5).mlu())

        # input.size(1) not divisible by \prod(kernel_size)

        fold = torch.nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
        with self.assertRaisesRegex(
            RuntimeError, r"be divisible by the product of kernel_size"
        ):
            fold(torch.randn(1, 5, 9).mlu())

        with self.assertRaisesRegex(
            RuntimeError, r"be divisible by the product of kernel_size"
        ):
            fold(torch.randn(1, 19, 9).mlu())

        # input.size(2) not matching the total number of sliding blocks

        with self.assertRaisesRegex(
            RuntimeError, r"match the calculated number of sliding blocks"
        ):
            fold = torch.nn.Fold(output_size=(4, 5), kernel_size=(2, 3))
            fold(torch.randn(1, 6, 10).mlu())

        with self.assertRaisesRegex(
            RuntimeError, r"match the calculated number of sliding blocks"
        ):
            fold = torch.nn.Fold(output_size=(4, 5), kernel_size=(2, 3), stride=(2, 2))
            fold(torch.randn(1, 6, 5).mlu())

        with self.assertRaisesRegex(
            RuntimeError, r"match the calculated number of sliding blocks"
        ):
            fold = torch.nn.Fold(
                output_size=(4, 5),
                kernel_size=(2, 3),
                stride=(2, 2),
                dilation=(1, 2),
                padding=(2, 0),
            )
            fold(torch.randn(1, 6, 5).mlu())  # should be 4 * 1 = 4 sliding blocks

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_nn_fold_bfloat16(self):
        # only support 3 dims or 4 dims input.
        N_lst = [0, 1, 5]
        Ci_lst = [1, 3]
        H_lst = [20, 32]
        W_lst = [22, 34]
        K_lst = [1, 3]
        padding_lst = [1, 2]
        stride_lst = [1, 4]
        dilation_lst = [1, 3]
        dtype_list = [torch.bfloat16]
        func_lst = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        product_list = product(
            N_lst,
            Ci_lst,
            H_lst,
            W_lst,
            K_lst,
            padding_lst,
            stride_lst,
            dilation_lst,
            dtype_list,
            func_lst,
        )
        for N, Ci, H, W, K, P, S, D, dtype, func in product_list:
            H_O = int(((H + 2 * P) - (D * (K - 1) + 1)) / S + 1)
            W_O = int(((W + 2 * P) - (D * (K - 1) + 1)) / S + 1)
            input_cpu, input_mlu = self.TensorGenerator(
                (N, Ci * K * K, H_O * W_O), dtype, func
            )
            input_cpu = torch.nn.Parameter(input_cpu)
            input_mlu = torch.nn.Parameter(input_mlu)
            fold = torch.nn.Fold(
                output_size=(H, W),
                kernel_size=(K, K),
                stride=(S, S),
                padding=(P, P),
                dilation=(D, D),
            )
            output_cpu = fold(input_cpu)
            output_mlu = fold(input_mlu)
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0.003, use_MSE=True)
            grad_cpu, grad_mlu = self.TensorGenerator(output_cpu.size(), dtype, func)
            output_cpu.backward(grad_cpu)
            output_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                input_cpu.grad, input_mlu.grad.cpu(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    unittest.main()
