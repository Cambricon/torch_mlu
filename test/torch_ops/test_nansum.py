from __future__ import print_function

import sys
import os
import itertools
import unittest
import logging
from itertools import product
import copy

import torch
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)

logging.basicConfig(level=logging.DEBUG)

# The sum operator uses the calculation result of double data type as the
# reference value, while the calculation error of float type is large.


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_nansum_dim(self):
        type_list = [True, False]
        shape_list = [
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
            (2, 0, 3),
        ]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len + 1):
                dim_lists = list(itertools.permutations(range(dim_len), i)) + list(
                    itertools.permutations(range(-dim_len, 0), i)
                )
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        index = torch.tensor([0])
                        x.index_fill_(0, index, float("nan"))
                        out_cpu = x.double().nansum(test_dim, keepdim=test_type).float()
                        out_mlu = self.to_device(x).nansum(test_dim, keepdim=test_type)
                        self.assertTensorsEqual(
                            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                        )

    # @unittest.skip("not test")
    @testinfo()
    def test_nansum(self):
        shape_list = [
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
            (2, 0, 3),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            index = torch.tensor([0])
            x.index_fill_(0, index, float("nan"))
            out_cpu = torch.nansum(x.double()).float()
            out_mlu = torch.nansum(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nansum_scalar(self):
        x = torch.tensor(float("nan"), dtype=torch.float)
        out_cpu = torch.nansum(x.double()).float()
        out_mlu = torch.nansum(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nansum_out(self):
        type_list = [True, False]
        shape_list = [
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
            (2, 0, 3),
        ]
        for shape in shape_list:
            dim_len = len(shape)
            for i in range(1, dim_len + 1):
                dim_lists = list(itertools.permutations(range(dim_len), i)) + list(
                    itertools.permutations(range(-dim_len, 0), i)
                )
                for test_dim in dim_lists:
                    for test_type in type_list:
                        x = torch.randn(shape, dtype=torch.float)
                        index = torch.tensor([0])
                        x.index_fill_(0, index, float("nan"))
                        out_cpu = torch.randn(1)
                        out_mlu = self.to_mlu(torch.randn(1))
                        x_mlu = self.to_mlu(x)
                        torch.nansum(
                            x.double(), test_dim, keepdim=test_type, out=out_cpu
                        )
                        torch.nansum(x_mlu, test_dim, keepdim=test_type, out=out_mlu)
                        try:
                            self.assertTensorsEqual(
                                out_cpu.float(),
                                out_mlu.cpu().float(),
                                0.003,
                                use_MSE=True,
                            )
                        except AssertionError as e:
                            # results of CPU and MLU are out of threshold and
                            # MLU, numpy, gpu results are same, so use numpy
                            # result to compare with MLU
                            print(e)
                            # use double to ensure precision
                            x_numpy = x.double().numpy()
                            out_sum = np.nansum(
                                x_numpy, axis=test_dim, keepdims=test_type
                            )
                            self.assertTensorsEqual(
                                torch.from_numpy(out_sum).float(),
                                out_mlu.cpu(),
                                0.003,
                                use_MSE=True,
                            )

    # @unittest.skip("not test")
    @testinfo()
    def test_nansum_dtype(self):
        shape = (2, 3, 4)
        type_list = [torch.int, torch.int16, torch.int8, torch.long]
        out_dtype_list = [
            torch.int8,
            torch.int16,
            torch.int,
            torch.half,
            torch.float,
            torch.double,
        ]
        for t in type_list:
            for out_dtype in out_dtype_list:
                x = torch.randn(shape, dtype=torch.float) * 10000
                index = torch.tensor([0])
                if t.is_floating_point:
                    x.index_fill_(0, index, float("nan")).to(t)
                # nansum only support floating types and bfloat16 on cpu
                if t.is_floating_point:
                    out_cpu = x.nansum(dim=1, keepdim=True, dtype=out_dtype)
                else:
                    out_cpu = x.sum(dim=1, keepdim=True, dtype=out_dtype)
                out_mlu = x.to("mlu").nansum(dim=1, keepdim=True, dtype=out_dtype)
                self.assertEqual(out_cpu.dtype, out_mlu.dtype)
                # TODO(hyl) cnnl cast unsupport int16->int8
                if t is not torch.int16 and out_dtype is not torch.int8:
                    self.assertTensorsEqual(
                        out_cpu.float(),
                        out_mlu.cpu().float(),
                        0.003,
                        use_MSE=True,
                        allow_inf=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_nansum_backward(self):
        keepdim_list = [True, False]
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        for shape in shape_list:
            dim_len = len(shape)
            for item in product(keepdim_list, range(-dim_len, dim_len)):
                x = torch.randn(shape, dtype=torch.float)
                index = torch.tensor([0])
                x.index_fill_(0, index, float("nan"))
                x.requires_grad_(True)
                x_mlu = self.to_device(x)

                out_cpu = torch.nansum(x, item[1], keepdim=item[0])
                grad = torch.randn(out_cpu.shape)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                out_cpu.backward(grad)
                x_grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()

                out_mlu = torch.nansum(x_mlu, item[1], keepdim=item[0])
                out_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    x_grad_cpu.float(), x.grad.float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_nansum_exception(self):
        x = torch.randn(3, 4, 5, device="mlu")
        y = torch.randn(3, device="mlu")
        msg = "Expected out tensor to have dtype c10::Half, but got float instead"
        with self.assertRaises(RuntimeError) as cm:
            _ = torch.sum(input=x, out=y, dim=0, dtype=torch.half)
        self.assertEqual(cm.exception.args[0], msg)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_nansum_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 10, 104), dtype=torch.bfloat16, device="cpu"
        )
        left_cpu = torch.nn.Parameter(left)
        left_mlu = torch.nn.Parameter(left.mlu())
        out_cpu = torch.nansum(left_cpu)
        out_mlu = torch.nansum(left_mlu)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = copy.deepcopy(grad).to("mlu")
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            left_cpu.grad.float(), left_mlu.grad.cpu().float(), 0.0, use_MSE=True
        )
        left_cpu.grad.zero_()
        left_mlu.grad.zero_()
        out_cpu = torch.nansum(left_cpu, 1, keepdim=True)
        out_mlu = torch.nansum(left_mlu, 1, keepdim=True)
        grad = torch.randn(out_cpu.shape)
        grad_mlu = copy.deepcopy(grad).to("mlu")
        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            left_cpu.grad.float(), left_mlu.grad.cpu().float(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
