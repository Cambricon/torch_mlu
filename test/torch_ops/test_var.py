from __future__ import print_function

import sys
import os
import itertools
import unittest
import logging
import copy
from itertools import product

import torch
from torch.testing import make_tensor

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)

logging.basicConfig(level=logging.DEBUG)


# The var operator calculates the variance for each row of the input tensor in a given
def to_non_dense_channels_last(data, dim=None, distance=2):
    if not type(data) == torch.Tensor:
        print(
            "[Warning]: It's not available to convert an unknown object to non-dense type"
        )
        return data
    # convert the last channel as default.
    convert_dim = data.dim()
    if dim is not None:
        convert_dim = dim
    if convert_dim > data.dim():
        print(
            f"[Warning]: The max available expand dim for a {data.dim()} Tensor"
            f" is {data.dim()}, but got specified dim as {dim}."
        )
        convert_dim = data.dim()
    a = data.unsqueeze(convert_dim)
    b = torch.cat([a for _ in range(distance)], convert_dim)
    return b.select(dim=convert_dim, index=0)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_var_dim(self):
        type_list = [True, False]
        shape_list = [
            (3, 2, 128, 10, 6),
            (2, 128, 10, 6),
            (200, 1536, 202),
            (2, 100),
            (24,),
        ]
        for shape in shape_list:
            dim_len = len(shape)
            dim_lists = list(range(dim_len))
            for test_dim in dim_lists:
                for test_type in type_list:
                    x = torch.randn(shape, dtype=torch.float)
                    out_cpu = torch.var(x, dim=test_dim, keepdim=test_type)
                    out_mlu = torch.var(x.mlu(), dim=test_dim, keepdim=test_type)
                    out_cpu_ = x.var(dim=test_dim, keepdim=test_type)
                    out_mlu_ = x.mlu().var(dim=test_dim, keepdim=test_type)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    self.assertTensorsEqual(
                        out_cpu_, out_mlu_.cpu(), 0.003, use_MSE=True
                    )

            dim_lists_neg = list(itertools.permutations(range(-dim_len, 0), 1))
            for test_dim in dim_lists_neg:
                for test_type in type_list:
                    x = torch.randn(shape, dtype=torch.float)
                    out_cpu = torch.var(x, dim=test_dim, keepdim=test_type)
                    out_mlu = torch.var(x.mlu(), dim=test_dim, keepdim=test_type)
                    out_cpu_ = x.var(dim=test_dim, keepdim=test_type)
                    out_mlu_ = x.mlu().var(dim=test_dim, keepdim=test_type)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    self.assertTensorsEqual(
                        out_cpu_, out_mlu_.cpu(), 0.003, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_var(self):
        shape_list = [(3, 2, 128, 10, 6), (2, 128, 10, 6), (2, 512, 8), (2, 100), (24,)]
        unbiased_list = [True, False]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            for unbiased in unbiased_list:
                # (TODO) guolin var only Calculates the variance
                # for each row of the input tensor in a given,not spport dims
                out_cpu = torch.var(x, dim=0, unbiased=unbiased)
                out_mlu = torch.var(self.to_mlu(x), dim=0, unbiased=unbiased)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_backward(self):
        shape = (3, 2, 128, 10, 6)
        x = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        out_cpu = torch.var(x, dim=2)
        grad = torch.randn(out_cpu.shape, dtype=torch.float32)
        out_cpu.backward(grad)
        x_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu = torch.var(self.to_mlu(x), dim=2)
        out_mlu.backward(self.to_mlu(grad))
        self.assertTensorsEqual(x_grad_cpu, x.grad, 0.003, use_MSE=True)

        x1 = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        out_cpu1 = torch.var(x1)
        out_cpu1.backward()
        x_grad_cpu1 = copy.deepcopy(x1.grad)
        x1.grad.zero_()
        out_mlu1 = torch.var(self.to_mlu(x1))
        out_mlu1.backward()
        self.assertTensorsEqual(x_grad_cpu1, x1.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_channel_last(self):
        shape_list = [(3, 2, 128, 10, 6), (2, 128, 10, 6), (2, 512, 8), (2, 100), (24,)]
        unbiased_list = [True, False]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            x_mlu = x.mlu()
            x = self.convert_to_channel_last(x)
            x_mlu = self.convert_to_channel_last(x_mlu)
            for unbiased in unbiased_list:
                # (TODO) guolin var only Calculates the variance for each
                # row of the input tensor in a given,not spport dims
                out_cpu = torch.var(x, dim=0, unbiased=unbiased)
                out_mlu = torch.var(x_mlu, dim=0, unbiased=unbiased)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_non_dense_channel_last(self):
        x = torch.randn(3, 5, 3, 3)
        out_cpu = torch.var(x, (-1,))
        out_mlu = torch.var(to_non_dense_channels_last(x.mlu(), 2), (-1,))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_no_contiguous(self):
        a = torch.randn(4, 4, dtype=torch.float)
        x = a[::2, ::2]
        out_cpu = torch.var(x, dim=0, keepdim=False)
        a = a.to("mlu")
        x = a[::2, ::2]
        out_mlu = torch.var(x, dim=0, keepdim=False)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_out(self):
        shape_list = [(3, 2, 128, 10, 6), (2, 128, 10, 6), (2, 512, 8), (2, 100), (24,)]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        out_cpu = torch.randn(1, dtype=torch.float32)
        out_mlu = out_cpu.mlu()
        for shape in shape_list:
            dim_len = len(shape)
            dim_lists = list(range(dim_len))
            x = torch.randn(shape, dtype=torch.float32)
            for unbiased in unbiased_list:
                for keepdim in keepdim_list:
                    for dim in dim_lists:
                        # (TODO) guolin var only Calculates the variance
                        # for each row of the input tensor in a given,not support dims
                        torch.var(
                            x, dim=dim, unbiased=unbiased, keepdim=keepdim, out=out_cpu
                        )
                        torch.var(
                            self.to_mlu(x),
                            dim=dim,
                            unbiased=unbiased,
                            keepdim=keepdim,
                            out=out_mlu,
                        )
                        self.assertTensorsEqual(
                            out_cpu, out_mlu.cpu(), 0.003, use_MSE=True
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_var_out_no_dense(self):
        x = torch.randn(5, 3, dtype=torch.float32)
        y = torch.randn(5, 3, dtype=torch.float32)
        x_mlu = x.mlu()
        y_mlu = y.mlu()
        torch.var(x, 1, keepdim=False, out=y[:, 1])
        torch.var(x_mlu, 1, keepdim=False, out=y_mlu[:, 1])
        self.assertTrue(y.stride() == y_mlu.stride())
        self.assertTensorsEqual(y[:, 1], y_mlu[:, 1].cpu(), 0.003, use_MSE=True)

    @unittest.skip("not test")
    @testinfo()
    # In current version, this test case cannot be passed due to the scenario where
    # var.out takes a Dimnames list as input dim is not  supported yet
    def test_var_out_all_no_dense(self):
        x = torch.randn(5, 3, dtype=torch.float32)
        y = torch.randn(5, 3, dtype=torch.float32)
        x_mlu = x.mlu()
        y_mlu = y.mlu()
        y = y[:, 1]
        y_mlu = y_mlu[:, 1]
        torch.var(x, [], keepdim=False, out=y)
        torch.var(x_mlu, [], keepdim=False, out=y_mlu)
        self.assertTrue(y.stride() == y_mlu.stride())
        self.assertTensorsEqual(y, y_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_empty_input(self):
        shape = (2, 0, 4)
        x = torch.randn(shape, dtype=torch.float32)
        keepdim_list = [True, False]
        unbiased_list = [True, False]
        # In this test case, empty tensors such as tensor([], device='mlu:0', size=(2, 0, 1)
        # or tensor([], device='mlu:0', size=(2, 0)) will be generated
        for keepdim, isUnbiased in itertools.product(keepdim_list, unbiased_list):
            out_cpu = torch.var(x, dim=2, keepdim=keepdim, unbiased=isUnbiased)
            out_mlu = torch.var(x.mlu(), dim=2, keepdim=keepdim, unbiased=isUnbiased)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=False)
        # In this test case, non-empty tensor filled with nan will be generated, such as
        # tensor([[[nan, nan, nan, nan]], [[nan, nan, nan, nan]]], device='mlu:0')
        for keepdim, isUnbiased in itertools.product(keepdim_list, unbiased_list):
            out_cpu = torch.var(x, dim=1, keepdim=keepdim, unbiased=isUnbiased)
            out_mlu = torch.var(x.mlu(), dim=1, keepdim=keepdim, unbiased=isUnbiased)
            self.assertEqual(out_cpu.shape, out_mlu.shape)
            self.assertTrue(torch.isnan(out_mlu).all())
        # In this test case, dim will not be provided, and the output should be an empty tensor
        # filled with nan like tensor(nan, device='mlu:0')
        for isUnbiased in unbiased_list:
            out_cpu = torch.var(x, unbiased=isUnbiased)
            out_mlu = torch.var(x.mlu(), unbiased=isUnbiased)
            self.assertEqual(out_cpu.shape, out_mlu.shape)
            self.assertTrue(torch.isnan(out_mlu).all())

    # @unittest.skip("not test")
    @testinfo()
    def test_var_negative_dims(self):
        shape_list = [
            (3, 2, 128, 10, 6),
            (2, 128, 10, 6),
            (200, 1536, 202),
            (2, 100),
            (24,),
        ]
        keepdim_list = [True, False]
        unbiased_list = [True, False]
        for items in itertools.product(shape_list, keepdim_list, unbiased_list):
            shape = items[0]
            keepdim = items[1]
            isUnbiased = items[2]
            for dim in range(-len(shape), -1):
                x = torch.randn(shape, dtype=torch.float32)
                out_cpu = torch.var(x, dim=dim, keepdim=keepdim, unbiased=isUnbiased)
                out_mlu = torch.var(
                    x.mlu(), dim=dim, keepdim=keepdim, unbiased=isUnbiased
                )
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_all(self):
        shape_list = [
            (3, 2, 128, 10, 6),
            (2, 128, 10, 6),
            (200, 1536, 202),
            (2, 100),
            (24,),
        ]
        keepdim_list = [True, False]
        unbiased_list = [True, False]
        for items in itertools.product(shape_list, keepdim_list, unbiased_list):
            shape = items[0]
            keepdim = items[1]
            isUnbiased = items[2]
            x = torch.randn(shape, dtype=torch.float32)
            out_cpu = torch.var(x, dim=[], keepdim=keepdim, unbiased=isUnbiased)
            out_mlu = torch.var(x.mlu(), dim=[], keepdim=keepdim, unbiased=isUnbiased)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            out_cpu = torch.var(x)
            out_mlu = torch.var(x.mlu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_backward(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x_mlu = self.to_device(x)

                out_cpu = torch.var(x, item[1], keepdim=item[0])
                grad = torch.randn(out_cpu.shape)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                out_cpu.backward(grad)
                x_grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()

                out_mlu = torch.var(x_mlu, item[1], keepdim=item[0])
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x.grad.float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_var_multidims(self):
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6)]
        dtype_list = [torch.float, torch.half]
        dim_list = [[0, 2], [1, 3], [0, 1, 3], [-1, -2, 0], None]
        unbiased_list = [True, False]
        keepdim_list = [True, False]
        for channels_last in [True, False]:
            for item in product(
                shape_list, dtype_list, dim_list, unbiased_list, keepdim_list
            ):
                x = torch.randn(item[0], dtype=item[1])
                if channels_last:
                    x = self.convert_to_channel_last(x)
                x_mlu = self.to_device(copy.deepcopy(x))
                x.requires_grad = True
                x_mlu.requires_grad = True

                out_cpu = torch.var(x, dim=item[2], unbiased=item[3], keepdim=item[4])
                grad = torch.randn(out_cpu.shape, dtype=item[1])
                if channels_last:
                    grad = self.convert_to_channel_last(grad)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                out_cpu.backward(grad)

                out_mlu = torch.var(
                    x_mlu, dim=item[2], unbiased=item[3], keepdim=item[4]
                )
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    x.grad.float(), x_mlu.grad.cpu().float(), 0.003, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("41GB")
    def test_var_large(self):
        shape_list = [(4, 1025, 1024 * 1024), (1, 4 * 1025 * 1024 * 1024)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float32)
            out_cpu = torch.var(x, dim=0)
            out_mlu = torch.var(self.to_mlu(x), dim=0)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_var_out_exceptions(self):
        ref_msg = r"result type Float can't be cast to the desired output type Long"
        x = torch.randn(5, 5, dtype=torch.float, device="mlu")
        res = torch.var(x)
        out = make_tensor(res.shape, dtype=torch.long, device="mlu")
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.var(x, out=out)

        ref_msg = r"now correction only supports 0 and 1 but got 2"
        x = torch.randn(5, 5, dtype=torch.float, device="mlu")
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.var(x, correction=2)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_var_bfloat16(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.bfloat16)
                x_mlu = self.to_device(x)
                out_cpu = torch.var(x, item[1], keepdim=item[0])
                out_mlu = torch.var(x_mlu, item[1], keepdim=item[0])
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
