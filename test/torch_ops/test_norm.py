from __future__ import print_function
import sys
import os
import unittest
import logging
import copy
from itertools import product
import torch
import torch_mlu  # pylint: disable=W0611

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
    skipBFloat16IfNotSupport,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_l1l2norm(self):
        shape_list = [(2899, 76), (2, 3, 4, 3, 2)]
        scalar_ops_list = [1, 2]
        dim_list = [-1, 0, 1, (0, 1)]
        keep_list = [True, False]
        loop_var = [shape_list, scalar_ops_list, dim_list, keep_list]
        for shape, scalar_op, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = x.norm(scalar_op, dim, keepdim=keep)
            out_mlu = self.to_mlu(x).norm(scalar_op, dim, keepdim=keep)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

        for shape, scalar_op, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = x.norm(scalar_op)
            out_mlu = self.to_device(x).norm(scalar_op)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

        # test fp16
        shape_list2 = [(2, 3, 4, 3, 2)]
        loop_var2 = [shape_list2, scalar_ops_list, dim_list, keep_list]
        for shape, scalar_op, dim, keep in product(*loop_var2):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = x.norm(scalar_op, dim, keepdim=keep)
            out_mlu = self.to_mlu_dtype(x, torch.half).norm(
                scalar_op, dim, keepdim=keep
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

        for shape, scalar_op, dim, keep in product(*loop_var2):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = x.norm(scalar_op)
            out_mlu = self.to_mlu_dtype(x, torch.half).norm(scalar_op)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_l1l2norm_dtype_mode(self):
        shape_list = [(2899, 76), (2, 3, 4, 3, 2)]
        scalar_ops_list = [1, 2]
        # only support calculate the mean of floating types
        type_list = [torch.float]
        loop_var = [shape_list, scalar_ops_list, type_list]
        for shape, scalar_op, type in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = torch.norm(x, p=scalar_op, dtype=type)
            out_mlu = torch.norm(self.to_mlu(x), p=scalar_op, dtype=type)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_l1l2norm_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        scalar_ops_list = [1, 2]
        data_types = [torch.float, torch.half]
        for scalar_op in scalar_ops_list:
            for data_type in data_types:
                out_cpu = x.norm(scalar_op)
                out_mlu = self.to_mlu_dtype(x, data_type).norm(scalar_op)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_norm_out(self):
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
            torch.norm(input=x, p=2.0, dim=dim, keepdim=keepdim, out=out)
            torch.norm(input=x.mlu(), p=2.0, dim=dim, keepdim=keepdim, out=out_mlu)
            self.assertTensorsEqual(
                out.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_norm_backward(self):
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

            out = torch.norm(input=x, p=2.0, dim=dim, keepdim=keepdim)
            grad_cpu = torch.randn(out.shape)
            out.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()

            grad_mlu = copy.deepcopy(grad_cpu).mlu()
            out_mlu = torch.norm(input=x_mlu, p=2.0, dim=dim, keepdim=keepdim)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                x_grad_cpu.float(), x.grad.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_norm_out_no_contiguous(self):
        x = torch.randn(3, 4, 5, 6).mlu()
        y = torch.randn(3, 4, 5, 6).mlu()
        torch.norm(input=x, out=y[:, :, 1, :], dim=2, p=1.0)
        y_expected = x.norm(dim=2, p=1.0)
        self.assertTensorsEqual(
            y[:, :, 1, :].cpu(), y_expected.cpu(), 0.0, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("28GB")
    def test_l1l2norm_large(self):
        # mlu support 2G-num in single-dim, but there may be a large cumulative error, like (4096*48*13725, 1)
        shape_list = [(4, 1024, 48, 13725)]
        scalar_ops_list = ["fro"]
        dim_list = [0]
        keep_list = [True]
        loop_var = [shape_list, scalar_ops_list, dim_list, keep_list]
        for shape, scalar_op, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.float)
            out_cpu = x.norm(scalar_op, dim, keepdim=keep)
            out_mlu = self.to_device(x).norm(scalar_op, dim, keepdim=keep)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_l1l2norm_bfloat16(self):
        shape_list2 = [(2899, 76), (2, 3, 4, 3, 2)]
        scalar_ops_list = [1, 2]
        dim_list = [-1, 0, 1, (0, 1)]
        keep_list = [True, False]
        loop_var2 = [shape_list2, scalar_ops_list, dim_list, keep_list]
        for shape, scalar_op, dim, keep in product(*loop_var2):
            x = torch.rand(shape, dtype=torch.bfloat16).float()
            out_cpu = x.norm(scalar_op, dim, keepdim=keep)
            out_mlu = self.to_mlu_dtype(x, torch.bfloat16).norm(
                scalar_op, dim, keepdim=keep
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.005, use_MSE=True
            )

        for shape, scalar_op, dim, keep in product(*loop_var2):
            x = torch.rand(shape, dtype=torch.bfloat16).float()
            out_cpu = x.norm(scalar_op)
            out_mlu = self.to_mlu_dtype(x, torch.bfloat16).norm(scalar_op)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.005, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_norm_backward_bfloat16(self):
        shape = (4, 5, 6)
        dim = (0, 1)
        input_dtype = torch.bfloat16
        keepdim = False

        x = torch.randn(shape, dtype=input_dtype).float()
        x_mlu = self.to_mlu_dtype(copy.deepcopy(x), input_dtype)
        x.requires_grad = True
        x_mlu.requires_grad = True

        out = torch.norm(input=x, p=2.0, dim=dim, keepdim=keepdim)
        grad_cpu = torch.randn(out.shape)
        out.backward(grad_cpu)
        x_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()

        grad_mlu = self.to_mlu_dtype(copy.deepcopy(grad_cpu), input_dtype)
        out_mlu = torch.norm(input=x_mlu, p=2.0, dim=dim, keepdim=keepdim)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            x_grad_cpu.float(), x_mlu.grad.cpu().float(), 0.005, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("28GB")
    def test_l1l2norm_large_bfloat16(self):
        shape_list = [(4, 1024, 48, 13725)]
        scalar_ops_list = ["fro"]
        dim_list = [0]
        keep_list = [True]
        loop_var = [shape_list, scalar_ops_list, dim_list, keep_list]
        for shape, scalar_op, dim, keep in product(*loop_var):
            x = torch.rand(shape, dtype=torch.bfloat16).float()
            out_cpu = x.norm(scalar_op, dim, keepdim=keep)
            out_mlu = self.to_mlu_dtype(x, torch.bfloat16).norm(
                scalar_op, dim, keepdim=keep
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
