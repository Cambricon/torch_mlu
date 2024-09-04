from __future__ import print_function
import sys
import os
import unittest
import logging
import copy
from itertools import product
import torch
import torch_mlu  # pylint: disable=W0611
from torch import linalg as LA

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestLAnorm(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_vector_norm(self):
        shape_list = [
            (0, 10),
            (0, 0),
            (10, 0, 10),
            (2, 100),
            (128, 64),
            (32, 10, 64),
            (2, 3, 4, 3, 2),
        ]
        vector_ords = [0, 0.9, 1, 2, 3, -0.5, -1, -2, -3]  # not support inf/-inf
        keep_list = [True, False]
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        loop_var = [shape_list, vector_ords, keep_list, dtype_list, func_list]
        for shape, ord, keep, dtype, func in product(*loop_var):
            for dim in [None] + list(range(len(shape))):
                x = torch.rand(shape, dtype=torch.float, requires_grad=True)
                x_mlu = copy.deepcopy(x)
                if (
                    x.numel() == 0
                    and (ord < 0.0 or ord == float("inf"))
                    and (dim is None or x.shape[dim] == 0)
                ):
                    # RuntimeError: linalg.vector_norm cannot compute the
                    # {ord} norm on an empty tensor because the operation
                    # does not have an identity
                    continue
                out_cpu = LA.vector_norm(func(x), ord, dim, keepdim=keep)
                out_mlu = LA.vector_norm(
                    func(x_mlu.to(dtype).to("mlu")), ord, dim, keepdim=keep
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                if x.numel() == 0 or ord == 0 or dtype == torch.half:
                    # "empty input" or "0-norm" will result in x.grad = None
                    continue
                grad_cpu = torch.rand(out_cpu.shape)
                grad_mlu = copy.deepcopy(grad_cpu).to(dtype).to("mlu")
                out_cpu.backward(grad_cpu)
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x.grad.float(), x_mlu.grad.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_matrix_norm(self):
        test_cases = [
            # input size, dim
            ((2, 100), (0, 1)),
            ((10, 10), (-2, -1)),
            ((10, 10), (0, 1)),
            ((10, 10), (1, 0)),
            ((10, 10, 10, 10), (2, 0)),
            ((10, 10, 10, 10), (-1, -2)),
            ((10, 10, 10, 10), (-1, -3)),
            ((10, 10, 10, 10), (-3, 2)),
            ((2, 3, 4, 3, 2), (4, 1)),
        ]
        # not support "nuc"/2/-2, cause mlu not support svdvals
        matrix_ords = [1, -1, float("inf"), float("-inf"), "fro"]
        keep_list = [True, False]
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for (shape, dim), ord, keep, dtype, func in product(
            test_cases, matrix_ords, keep_list, dtype_list, func_list
        ):
            x = torch.rand(shape, dtype=torch.float, requires_grad=True)
            x_mlu = copy.deepcopy(x)
            out_cpu = LA.matrix_norm(func(x), ord, dim, keepdim=keep)
            out_mlu = LA.matrix_norm(
                func(x_mlu.to(dtype).to("mlu")), ord, dim, keepdim=keep
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            if dtype == torch.half:
                continue
            grad_cpu = torch.rand(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad_cpu).to(dtype).to("mlu")
            out_cpu.backward(grad_cpu)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                x.grad.float(), x_mlu.grad.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_LA_norm(self):
        # ##################### test vector norm ########################
        shape_list = [(2, 100), (128, 64), (32, 10, 64), (2, 3, 4, 3, 2)]
        vector_ords = [0.5, 1, 2, 3.5, -0.5, -1, -2, -3.5]
        dim_list = [-1, 0, 1]
        keep_list = [True, False]
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x: x, self.to_non_dense]
        loop_var = [shape_list, vector_ords, dim_list, keep_list, dtype_list, func_list]
        for shape, ord, dim, keep, dtype, func in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float, requires_grad=True)
            x_mlu = copy.deepcopy(x)
            out_cpu = LA.norm(func(x), ord, dim, keepdim=keep)
            out_mlu = LA.norm(func(x_mlu.to(dtype).to("mlu")), ord, dim, keepdim=keep)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            if dtype == torch.half:
                continue
            grad_cpu = torch.rand(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad_cpu).to(dtype).to("mlu")
            out_cpu.backward(grad_cpu)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                x.grad.float(), x_mlu.grad.cpu().float(), 0.003, use_MSE=True
            )
        # for ord=None and dim=None, x will be flattened to 1D
        # and the 2-norm of the resulting vector will be computed
        loop_var2 = [shape_list, keep_list, dtype_list, func_list]
        for shape, keep, dtype, func in product(*loop_var2):
            x = torch.rand(shape, dtype=torch.float, requires_grad=True)
            x_mlu = copy.deepcopy(x)
            out_cpu = LA.norm(func(x), keepdim=keep)
            out_mlu = LA.norm(func(x_mlu.to(dtype).to("mlu")), keepdim=keep)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            if dtype == torch.half:
                continue
            grad_cpu = torch.rand(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad_cpu).to(dtype).to("mlu")
            out_cpu.backward(grad_cpu)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                x.grad.float(), x_mlu.grad.cpu().float(), 0.003, use_MSE=True
            )

        # ##################### test matrix norm ########################
        shape_list = [(2, 100), (128, 64), (32, 10, 64), (2, 3, 4, 3, 2)]
        matrix_ords = [1, -1, float("inf"), float("-inf"), "fro"]
        dim_list = [(0, 1), (-2, -1), (1, 0)]
        keep_list = [True, False]
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x: x, self.to_non_dense]
        loop_var = [shape_list, matrix_ords, dim_list, keep_list, dtype_list, func_list]
        for shape, ord, dim, keep, dtype, func in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float, requires_grad=True)
            x_mlu = copy.deepcopy(x)
            out_cpu = LA.norm(func(x), ord, dim, keepdim=keep)
            out_mlu = LA.norm(func(x_mlu.to(dtype).to("mlu")), ord, dim, keepdim=keep)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
            if dtype == torch.half:
                continue
            grad_cpu = torch.rand(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad_cpu).to(dtype).to("mlu")
            out_cpu.backward(grad_cpu)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                x.grad.float(), x_mlu.grad.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_LA_norm_out(self):
        # ##################### test vector norm ########################
        shape_list = [(2, 100), (128, 64), (32, 10, 64), (2, 3, 4, 3, 2)]
        vector_ords = [0.5, 1, 2, 3.5, -0.5, -1, -2, -3.5]
        dim_list = [-1, 0, 1]
        keep_list = [True, False]
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x: x, self.to_non_dense]
        loop_var = [shape_list, vector_ords, dim_list, keep_list, dtype_list, func_list]
        for shape, ord, dim, keep, dtype, func in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = torch.empty(0, dtype=torch.float)
            out_mlu = out_cpu.to(dtype).to("mlu")
            LA.norm(func(x), ord, dim, keepdim=keep, out=out_cpu)
            LA.norm(func(x.to(dtype).to("mlu")), ord, dim, keepdim=keep, out=out_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
        # for ord=None and dim=None, x will be flattened to 1D
        # and the 2-norm of the resulting vector will be computed
        loop_var2 = [shape_list, keep_list, dtype_list, func_list]
        for shape, keep, dtype, func in product(*loop_var2):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = torch.empty(0, dtype=torch.float)
            out_mlu = out_cpu.to(dtype).to("mlu")
            LA.norm(func(x), keepdim=keep, out=out_cpu)
            LA.norm(func(x.to(dtype).to("mlu")), keepdim=keep, out=out_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

        # ##################### test matrix norm ########################
        shape_list = [(2, 100), (128, 64), (32, 10, 64), (2, 3, 4, 3, 2)]
        matrix_ords = [1, -1, float("inf"), float("-inf"), "fro"]
        dim_list = [(-1, -2), (-2, -1)]
        keep_list = [True, False]
        dtype_list = [torch.float, torch.half]
        func_list = [lambda x: x, self.to_non_dense]
        loop_var = [shape_list, matrix_ords, dim_list, keep_list, dtype_list, func_list]
        for shape, ord, dim, keep, dtype, func in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = torch.empty(
                shape[:-2] if not keep else shape[:-2] + (1, 1), dtype=torch.float
            )
            out_mlu = out_cpu.to(dtype).to("mlu")
            LA.norm(func(x), ord, dim, keepdim=keep, out=out_cpu)
            LA.norm(func(x.to(dtype).to("mlu")), ord, dim, keepdim=keep, out=out_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
