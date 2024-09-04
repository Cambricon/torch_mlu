import sys
import os
import unittest
import logging
import copy
from itertools import product
import torch
from torch import linalg as LA
import torch_mlu  # pylint: disable=W0611

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


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_l1_l2_vector_norm(self):
        shape_list = [(2899, 76), (2, 3, 4, 3, 2)]
        ord_list = [1, 2]
        dim_list = [-1, 0, 1, (0, 1)]
        keep_list = [True, False]
        loop_var = [shape_list, ord_list, dim_list, keep_list]
        for shape, ord, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = LA.vector_norm(x, ord=ord, dim=dim, keepdim=keep)
            out_mlu = LA.vector_norm(self.to_mlu(x), ord=ord, dim=dim, keepdim=keep)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )
        # test fp16
        shape_list2 = [(2, 3, 4, 3, 2)]
        loop_var2 = [shape_list2, ord_list, dim_list, keep_list]
        for shape, ord, dim, keep in product(*loop_var2):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = LA.vector_norm(x, ord=ord, dim=dim, keepdim=keep)
            out_mlu = LA.vector_norm(
                self.to_mlu_dtype(x, torch.half), ord=ord, dim=dim, keepdim=keep
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_vector_norm_dtype_mode(self):
        shape_list = [(2899, 76), (2, 3, 4, 3, 2)]
        ord_list = [1, 2]
        # only support calculate the mean of floating types
        type_list = [torch.float]
        loop_var = [shape_list, ord_list, type_list]
        for shape, ord, dtype in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = LA.vector_norm(x, ord=ord, dtype=dtype)
            out_mlu = LA.vector_norm(self.to_mlu(x), ord=ord, dtype=dtype)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_scalar_vector_norm(self):
        x = torch.tensor(5.2, dtype=torch.float)
        ord_list = [1, 2]
        data_types = [torch.float, torch.half]
        loop_var = [ord_list, data_types]
        for ord, dtype in product(*loop_var):
            out_cpu = LA.vector_norm(x, ord=ord)
            out_mlu = LA.vector_norm(self.to_mlu_dtype(x, dtype), ord=ord)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_vector_norm_out(self):
        input_shape_list = [(4, 5, 6), (5, 5, 149, 149)]
        dim_list = [(0, 1), (1, 2), (0, 2)]
        # TODO(sifengyang): diff of cpu/cuda norm.out accuracy in half dtype greats than 0.03.
        input_dtypes = [torch.float32, torch.double]
        keepdims = [False, True]
        for shape, dim, input_dtype, keepdim in product(
            input_shape_list, dim_list, input_dtypes, keepdims
        ):
            out = torch.randn(3, dtype=input_dtype)
            out_mlu = out.mlu()
            x = torch.randn(shape, dtype=input_dtype)
            LA.vector_norm(x, ord=2.0, dim=dim, keepdim=keepdim, out=out)
            LA.vector_norm(x.mlu(), ord=2.0, dim=dim, keepdim=keepdim, out=out_mlu)
            self.assertTensorsEqual(
                out.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_vector_norm_backward(self):
        input_shape_list = [(4, 5, 6), (5, 5, 149, 149)]
        dim_list = [(0, 1), (1, 2), (0, 2)]
        # TODO(sifengyang): diff of cpu/cuda norm.out accuracy in half dtype greats than 0.03.
        input_dtypes = [torch.float32, torch.double]
        keepdims = [False, True]
        for shape, dim, input_dtype, keepdim in product(
            input_shape_list, dim_list, input_dtypes, keepdims
        ):
            x = torch.randn(shape, dtype=input_dtype, requires_grad=True)
            x_mlu = x.mlu()

            out = LA.vector_norm(x, ord=2.0, dim=dim, keepdim=keepdim)
            grad_cpu = torch.randn(out.shape)
            out.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()

            grad_mlu = copy.deepcopy(grad_cpu).mlu()
            out_mlu = LA.vector_norm(x_mlu, ord=2.0, dim=dim, keepdim=keepdim)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                x_grad_cpu.float(), x.grad.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_vector_norm_out_non_contiguous(self):
        x = torch.randn(3, 4, 5, 6).mlu()
        y = torch.randn(3, 4, 5, 6).mlu()
        LA.vector_norm(x, out=y[:, :, 1, :], dim=2, ord=1.0)
        y_expected = LA.vector_norm(x, dim=2, ord=1.0)
        self.assertTensorsEqual(
            y[:, :, 1, :].cpu(), y_expected.cpu(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_norm_exceptions(self):
        x = torch.randn(4, 4, dtype=torch.float)
        msg = "torch_mlu does not support inf-Norm as p=inf/-inf."
        with self.assertRaises(RuntimeError) as cm:
            _ = LA.vector_norm(x.to("mlu"), ord=float("inf"))
        self.assertEqual(cm.exception.args[0], msg)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("28GB")
    def test_vector_norm_large(self):
        # mlu support 2G-num in single-dim, but there may be a large cumulative error, like (4096*48*13725, 1)
        shape_list = [(4, 1024, 48, 13725)]
        dim_list = [0]
        keep_list = [True]
        loop_var = [shape_list, dim_list, keep_list]
        for shape, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = LA.vector_norm(x, ord=2.0, dim=dim, keepdim=keep)
            out_mlu = LA.vector_norm(self.to_device(x), ord=2.0, dim=dim, keepdim=keep)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
