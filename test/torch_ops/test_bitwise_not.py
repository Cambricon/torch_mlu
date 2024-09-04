from __future__ import print_function

import sys
import os
import unittest
import logging
import random as rd
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
)

logging.basicConfig(level=logging.DEBUG)


class TestBitwiseNotOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_not(self):
        shape_list = [(10,), (2, 2, 3), (2, 0, 3), (2, 3, 4, 5)]
        dtype_list = [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.long,
        ]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype == torch.bool:
                    a = torch.randint(2, shape).type(dtype)
                elif dtype == torch.uint8:
                    a = torch.randint(16777216, shape).type(dtype)
                else:
                    a = torch.randint(-16777216, 16777216, shape).type(dtype)
                result_cpu = a.bitwise_not()
                result_mlu = self.to_device(a).bitwise_not()
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            # test scalar input
            if dtype == torch.bool:
                v = rd.choice([True, False])
            elif dtype == torch.uint8:
                v = rd.randint(0, 16777216)
            else:
                v = rd.randint(-16777216, 16777216)
            a = torch.tensor(v).type(dtype)
            result_cpu = a.bitwise_not()
            result_mlu = self.to_device(a).bitwise_not()
            self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)

            if a.dim() == 4:
                # test channel last
                a_in = a.to(memory_format=torch.channels_last)
                result_mlu2 = self.to_device(a_in).bitwise_not()
                self.assertTensorsEqual(result_cpu, result_mlu2.cpu(), 0)

                # test no dense
                a_in = a[:, :, :, :2]
                a_in_mlu = self.to_device(a_in)
                result_cpu_no_dense = a.bitwise_not()
                result_mlu_no_dense = a_in_mlu.bitwise_not()
                self.assertTrue(a_in.stride() == a_in_mlu.stride())
                self.assertTensorsEqual(
                    result_cpu_no_dense, result_mlu_no_dense.cpu(), 0
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_not_inplace(self):
        shape_list = [(5,), (3, 1), (2, 0, 3), (5, 4, 3, 2)]
        dtype_list = [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.long,
        ]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype == torch.bool:
                    a = torch.randint(2, shape).type(dtype)
                elif dtype == torch.uint8:
                    a = torch.randint(16777216, shape).type(dtype)
                else:
                    a = torch.randint(-16777216, 16777216, shape).type(dtype)
                a_mlu = self.to_device(a)
                a.bitwise_not_()
                raw_ptr = a_mlu.data_ptr()
                a_mlu.bitwise_not_()
                cur_ptr = a_mlu.data_ptr()
                self.assertEqual(raw_ptr, cur_ptr)
                self.assertTensorsEqual(a, a_mlu.cpu(), 0)

            # test scalar input
            if dtype == torch.bool:
                v = rd.choice([True, False])
            elif dtype == torch.uint8:
                v = rd.randint(0, 16777216)
            else:
                v = rd.randint(-16777216, 16777216)
            a = torch.tensor(v).type(dtype)
            a_mlu = self.to_device(a)
            a.bitwise_not_()
            raw_ptr = a_mlu.data_ptr()
            a_mlu.bitwise_not_()
            cur_ptr = a_mlu.data_ptr()
            self.assertEqual(raw_ptr, cur_ptr)
            self.assertTensorsEqual(a, a_mlu.cpu(), 0)

            if a.dim() == 4:
                # test channels last
                a = torch.tensor(v).type(dtype)
                a_in = a.to(memory_format=torch.channels_last)
                a_in_mlu = self.to_mlu(a_in)
                a_in.bitwise_not_()
                raw_ptr = a_in_mlu.data_ptr()
                a_in_mlu.bitwise_not_()
                cur_ptr = a_in_mlu.data_ptr()
                self.assertEqual(raw_ptr, cur_ptr)
                self.assertTensorsEqual(a_in, a_in_mlu.cpu(), 0)

                # test no dense
                a = torch.tensor(v).type(dtype)
                a_in = a[:, :, :, :2]
                a_in.bitwise_not_()
                a_in_mlu = self.to_mlu(a_in)
                raw_ptr = a_mlu.data_ptr()
                a_in_mlu.bitwise_not_()
                cur_ptr = a_mlu.data_ptr()
                self.assertTrue(a_in.stride() == a_in_mlu.stride())
                self.assertEqual(raw_ptr, cur_ptr)
                self.assertTensorsEqual(a_in, a_in_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_not_out(self):
        shape_list = [(10,)]
        dtype_list = [torch.bool]
        for dtype in dtype_list:
            for shape in shape_list:
                # the element number of out >= the expected of the op
                a = torch.randint(2, shape).type(dtype)
                res_cpu = torch.randint(2, shape).type(dtype)
                res_mlu = self.to_device(torch.randint(2, shape).type(dtype))
                torch.bitwise_not(a, out=res_cpu)
                torch.bitwise_not(self.to_device(a), out=res_mlu)
                self.assertTensorsEqual(res_cpu, res_mlu.cpu(), 0)
                # the element number of out < the expected of the op
                a = torch.randint(2, shape).type(dtype)
                res_cpu = torch.randint(2, (1,)).type(dtype)
                res_mlu = self.to_device(torch.randint(2, (1,)).type(dtype))
                torch.bitwise_not(a, out=res_cpu)
                torch.bitwise_not(self.to_device(a), out=res_mlu)
                self.assertTensorsEqual(res_cpu, res_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_not_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randint(2, shape_list[i]).type(torch.bool)
            out = torch.randint(2, shape_list[i]).type(torch.bool)
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.bitwise_not(x, out=out)
            torch.bitwise_not(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_bitwise_not_exception(self):
        shape = (10,)
        a = torch.randn(shape)
        res_mlu = self.to_device(torch.randn(shape))
        msg = r"MLU bitwise_not don't support tensor dtype Float."
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.bitwise_not(self.to_device(a), out=res_mlu)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_bitwise_not_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        dtype_list = [torch.bool, torch.int8]
        for dtype in dtype_list:
            for shape in shape_list:
                if dtype == torch.bool:
                    a = torch.randint(2, shape).type(dtype)
                else:
                    a = torch.randint(-16777216, 16777216, shape).type(dtype)
                result_cpu = a.bitwise_not()
                result_mlu = self.to_device(a).bitwise_not()
                self.assertTensorsEqual(result_cpu, result_mlu.cpu(), 0)


if __name__ == "__main__":
    run_tests()
