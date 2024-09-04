from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import copy
import torch

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


class TestArgMaxOp(TestCase):
    def check_result(self, x, ind_cpu, ind_mlu, dim):
        # max sorting algorithm for mlu is different from cpu,
        # when the max result has multi-

        if dim is None:
            # None dim means index is one number for max-value in full storage.
            x_tr = x.view(-1)
            ind_cpu_tr = ind_cpu.view(-1)
            ind_mlu_tr = ind_mlu.view(-1)
            t = None
        else:
            # the follow transpose and reshape will move the dim(reduce dim)
            # to the first of shape, and reshape it as [dim_size, other_size]
            # and then the arange t will select max-value due to the index,
            # so we can check if mlu and cpu choose the same max-value.
            x_tr = x.transpose(dim, 0).reshape(x.shape[dim], -1)
            ind_cpu_tr = ind_cpu.transpose(dim, 0).reshape(1, -1)
            ind_mlu_tr = ind_mlu.transpose(dim, 0).reshape(1, -1)
            t = torch.arange(0, x_tr.shape[1])
        self.assertTensorsEqual(
            x_tr[ind_cpu_tr[0, t], t], x_tr[ind_mlu_tr[0, t], t], 0.0
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_argmax(self):
        dtype_list = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.long,
            torch.float,
            torch.half,
            torch.double,
        ]
        shape_list = [
            (2, 3, 4),
            (10, 11, 9, 8),
            (32,),
            (15, 16, 8, 9, 10, 11),
            (2, 3, 4, 5, 6, 7, 8),
            (1, 256),
            (1, 1, 1),
        ]
        # [CNNLCORE-13417]dim value is none, cnnlGetReduceOpWorkspaceSize/cnnlReduce output dim <= input dim check error
        # pt1.9 output scalar, but pt1.13 output shape dependency keepdim parameter.
        # dim_list = [1, -1, 0, 2, 3, 6, 7, None]
        dim_list = [1, -1, 0, 2, 3, 6, 7]
        keepdim_choices = [True, False]
        mode_list = [self.to_non_dense, lambda x: x]
        list_list = [dtype_list, shape_list, dim_list, keepdim_choices, mode_list]
        for dtype, shape, dim, keepdim, mode in product(*list_list):
            x = torch.randn(shape)
            if dtype == torch.int:
                x = torch.randint(-10, 10, shape)
            x = x.to(dtype)
            if dim is not None and dim >= len(shape):
                dim = dim % len(shape)
            out_cpu = torch.argmax(mode(x), dim, keepdim=keepdim)
            out_mlu = torch.argmax(mode(self.to_device(x)), dim, keepdim=keepdim)
            out_mlu = out_mlu.cpu()
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertEqual(out_cpu.size(), out_mlu.size())
            if (not keepdim) and (dim is not None):
                out_cpu = out_cpu.unsqueeze(dim)
                out_mlu = out_mlu.unsqueeze(dim)
            if dtype == torch.half:
                x = x.to(torch.float)
            self.check_result(x, out_cpu, out_mlu, dim)

    # @unittest.skip("not test")
    @testinfo()
    def test_argmax_out(self):
        input_dtype_list = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.half,
            torch.float,
            torch.double,
        ]
        outptu_dtype_list = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.half,
            torch.float,
            torch.double,
        ]
        # [CNNLCORE-13417]dim value is none, cnnlGetReduceOpWorkspaceSize/cnnlReduce output dim <= input dim check error
        # pt1.9 output scalar, but pt1.13 output shape dependency keepdim parameter.
        # dim_list = [1, -1, 0, 2, 3, 6, 7, None]
        dim_list = [1, -1, 0, 2, 3, 6, 7]
        input_shapes = [(2, 3, 4), (3, 4, 5, 6), (4, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8)]
        keepdims = [True, False]
        params_list = [
            input_dtype_list,
            outptu_dtype_list,
            dim_list,
            keepdims,
            input_shapes,
        ]
        for in_dtype, out_dtype, dim, keepdim, input_shape in product(*params_list):
            y = torch.randn(3).to(torch.long)
            y_mlu = y.mlu()
            if in_dtype.is_floating_point:
                x = torch.randn(input_shape).to(in_dtype)
            else:
                x = torch.randint(-10, 10, input_shape).to(in_dtype)
            if dim is not None and dim >= len(input_shape):
                dim = dim % len(input_shape)
            output_cpu = torch.argmax(input=x, dim=dim, keepdim=keepdim, out=y)
            output_mlu = torch.argmax(
                input=x.mlu(), dim=dim, keepdim=keepdim, out=y_mlu
            )
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), 0.0)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_argmax_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 10, 24), dtype=torch.bfloat16, device="cpu"
        )
        left_mlu = left.mlu()
        out_cpu = torch.argmax(left, 1, keepdim=True)
        out_mlu = torch.argmax(left_mlu, 1, keepdim=True)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        output_cpu = torch.testing.make_tensor(
            out_cpu.shape, dtype=torch.long, device="cpu"
        )
        output_mlu = output_cpu.mlu()
        torch.argmax(left, 1, keepdim=True, out=output_cpu)
        torch.argmax(left_mlu, 1, keepdim=True, out=output_mlu)
        self.assertTensorsEqual(output_mlu.cpu(), output_cpu, 0.0, use_MSE=True)
        self.assertTensorsEqual(output_mlu.cpu(), out_mlu.cpu(), 0.0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("36GB")
    def test_argmax_large(self):
        dtype_list = [torch.int8, torch.float, torch.half]
        shape_list = [(48, 4096, 13725), (1, 4096 * 48 * 13725)]
        dim_list = [0]
        keepdim_choices = [True]
        list_list = [dtype_list, shape_list, dim_list, keepdim_choices]
        for dtype, shape, dim, keepdim in product(*list_list):
            x = torch.randn(shape)
            if dtype == torch.int:
                x = torch.randint(-10, 10, shape)
            x = x.to(dtype)
            if dim is not None and dim >= len(shape):
                dim = dim % len(shape)
            out_cpu = torch.argmax(x, dim, keepdim=keepdim)
            out_mlu = torch.argmax(self.to_device(x), dim, keepdim=keepdim)
            out_mlu = out_mlu.cpu()
            self.assertEqual(out_cpu.dtype, out_mlu.dtype)
            self.assertEqual(out_cpu.size(), out_mlu.size())


if __name__ == "__main__":
    run_tests()
