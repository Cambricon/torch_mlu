from __future__ import print_function

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
    TEST_BFLOAT16,
    TestCase,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestTransposeOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_transpose(self):
        shape_list = [
            (126, 24, 1024),
            (4, 12, 45, 100),
            (4, 5, 6, 7, 8),
            (3, 4, 10, 200, 10, 20),
        ]
        dim0_lst = [0, 1, 2]
        dim1_lst = [0, 1, 2]
        data_types = [(torch.float, 0.0), (torch.half, 0.0)]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func, dim0, dim1 in product(
            shape_list, data_types, func_list, dim0_lst, dim1_lst
        ):
            dtype, err = dtype_err
            x = torch.randn(shape, dtype=dtype)
            x_mlu = x.mlu()
            output_cpu = func(x).transpose(dim0, dim1)
            output_mlu = func(x_mlu).transpose(dim0, dim1)
            self.assertTensorsEqual(output_cpu.float(), output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_transpose_(self):
        shape_list = [
            (126, 24, 1024),
            (4, 12, 45, 100),
            (4, 5, 6, 7, 8),
            (3, 4, 10, 200, 10, 20),
        ]
        dim0_lst = [0, 1, 2]
        dim1_lst = [0, 1, 2]
        data_types = [(torch.float, 0.0), (torch.half, 0.0)]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype_err, func, dim0, dim1 in product(
            shape_list, data_types, func_list, dim0_lst, dim1_lst
        ):
            dtype, err = dtype_err
            x = func(torch.randn(shape, dtype=dtype))
            x_mlu = func(x.mlu())
            ori_ptr = x_mlu.data_ptr()
            x.transpose_(dim0, dim1)
            x_mlu.transpose_(dim0, dim1)
            self.assertTrue(ori_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_t(self):
        shape_lst = [(3, 44), (6, 123), (45, 100), (23), ()]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for data_type, err in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                output_cpu = x.t()
                output_mlu = x_mlu.t()
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_t_inplace(self):
        shape_lst = [(3, 44), (6, 123), (45, 100), (23), ()]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for data_type, err in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                ori_ptr = x_mlu.data_ptr()
                x.t_()
                x_mlu.t_()
                self.assertTrue(ori_ptr, x_mlu.data_ptr())
                self.assertTensorsEqual(x, x_mlu.cpu().float(), err)

    # @unittest.skip("not test")
    @testinfo()
    def test_t_exception(self):
        x = torch.randn(1, 2, 3, dtype=torch.float).to("mlu")
        ref_msg = "t\(\) expects a tensor with <= 2 dimensions, but self is 3D"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output_mlu = x.t()

    # @unittest.skip("not test")
    @testinfo()
    def test_T(self):
        shape_lst = [(3, 44), (6, 123), (45, 100), (23), ()]
        data_types = [(torch.float, 0.0), (torch.half, 3e-3)]
        for shape in shape_lst:
            for data_type, err in data_types:
                x = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x, data_type)
                output_cpu = x.T
                output_mlu = x_mlu.T
                self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), err)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_transpose_bfloat16(self):
        input_t = torch.rand((2, 2, 24, 3), dtype=torch.bfloat16)
        input_cpu = torch.nn.Parameter(input_t)
        input_mlu = torch.nn.Parameter(input_t.mlu())
        out_cpu = input_cpu.transpose(1, 3)
        out_mlu = input_mlu.transpose(1, 3)
        grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
        out_cpu.backward(grad)
        out_mlu.backward(grad.mlu())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0, use_MSE=True)
        self.assertTensorsEqual(input_cpu.grad, input_mlu.grad.cpu(), 0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
