from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    TEST_BFLOAT16,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestNonzeroOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_nonzero(self):
        shape_list = [(10,), (2, 2, 3), (2, 0, 3), (2, 3, 4, 5), (2, 3, 4, 5, 6)]
        dtype_list = [
            torch.bool,
            torch.float32,
            torch.int32,
            torch.double,
            torch.long,
            torch.uint8,
            torch.int8,
            torch.int16,
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype, func in product(shape_list, dtype_list, func_list):
            x = func(torch.randn(shape).to(dtype))
            x_mlu = x.to("mlu")
            result_cpu = torch.nonzero(x, as_tuple=False)
            result_mlu = torch.nonzero(x_mlu, as_tuple=False)
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_nonzero_out(self):
        shape_list = [(2, 2, 3), (2, 0, 3), (2, 3, 4, 5), (2, 3, 4, 5, 6)]
        dtype_list = [
            torch.bool,
            torch.float32,
            torch.int32,
            torch.double,
            torch.long,
            torch.uint8,
            torch.int8,
            torch.int16,
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype, func in product(shape_list, dtype_list, func_list):
            # the element number of out >= the expected of the op
            # multiply by 10, since torch.randn() may generate zero tensor when cast to int32.
            x = func((torch.randn(shape) * 10).to(dtype))
            x_mlu = x.to("mlu")
            out_cpu = func(torch.randn(x.numel() * x.dim()).to(torch.long))
            out_mlu = copy.deepcopy(out_cpu).to("mlu")
            # print(x)
            ori_ptr = out_mlu.data_ptr()
            ori_ptr_cpu = out_cpu.data_ptr()
            torch.nonzero(x, out=out_cpu)
            torch.nonzero(x_mlu, out=out_mlu)
            self.assertEqual(ori_ptr_cpu, out_cpu.data_ptr())
            self.assertEqual(ori_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

            # resize output
            # the element number of out < the expected of the op
            out_cpu_1 = func(torch.randn([]).to(torch.long))
            out_mlu_1 = copy.deepcopy(out_cpu_1).to("mlu")
            torch.nonzero(x, out=out_cpu_1)
            torch.nonzero(x_mlu, out=out_mlu_1)
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nonzero_as_tuple(self):
        shape_list = [(10,), (2, 2, 3), (2, 0, 3), (2, 3, 4, 5), (2, 3, 4, 5, 6)]
        dtype_list = [
            torch.bool,
            torch.float32,
            torch.int32,
            torch.double,
            torch.long,
            torch.uint8,
            torch.int8,
            torch.int16,
        ]
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        for shape, dtype, func in product(shape_list, dtype_list, func_list):
            x = func(torch.randn(shape).to(dtype))
            x_mlu = x.to("mlu")
            result_cpu = torch.nonzero(x, as_tuple=True)
            result_mlu = torch.nonzero(x_mlu, as_tuple=True)
            self.assertEqual(len(result_mlu), len(result_cpu))
            for res_cpu, res_mlu in zip(result_cpu, result_mlu):
                self.assertTensorsEqual(res_cpu, res_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_nonzero_scalar(self):
        ## test scalar input
        a = torch.tensor(0).type(torch.int8)
        result_cpu = torch.nonzero(a, as_tuple=False)
        result_mlu = torch.nonzero(self.to_device(a), as_tuple=False)
        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

        a = torch.tensor(1).type(torch.bool)
        out_cpu = torch.randint(3, (1,))
        out_mlu = self.to_device(torch.randint(3, (1,)))
        torch.nonzero(a, out=out_cpu)
        torch.nonzero(self.to_device(a), out=out_mlu)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_nonzero_expetion(self):
        shape = (2, 2, 3)
        a = torch.randint(3, shape).type(torch.int8).to("mlu")
        out_mlu = self.to_device(
            torch.randn(
                (a.numel() * a.dim()),
            )
        )
        ref_msg = "the datatype of out in cnnl_nonzero_out must be Long, but got Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.nonzero(a, out=out_mlu)

        shape = (2, 2, 3, 1, 2, 1, 3, 2, 1)
        a = torch.randint(3, shape).type(torch.int8).to("mlu")
        out_mlu = self.to_device(torch.randn((a.numel() * a.dim())).to(torch.long))
        ref_msg = "nonzero is not supported for tensor with more than 8 dimensions"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.nonzero(a, out=out_mlu)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_nonzero_large(self):
        shape_list = [(5, 1024, 1024, 1024), (4294967296, 1)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = x.to("mlu")
            ref_msg = "nonzero is not supported for tensors with more than INT_MAX elements, file a support request"
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                out_mlu = torch.nonzero(x_mlu)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_nonzero_bfloat16(self):
        x = torch.randn((2, 3, 4, 5, 6)).to(torch.bfloat16)
        x_mlu = x.to("mlu")
        result_cpu = torch.nonzero(x, as_tuple=False)
        result_mlu = torch.nonzero(x_mlu, as_tuple=False)
        self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)


if __name__ == "__main__":
    unittest.main()
