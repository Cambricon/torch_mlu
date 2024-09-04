from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
from scipy import stats

import torch

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
torch.manual_seed(2)


class TestNormal_Op(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_normal_orig(self):  # pylint: disable=R0201
        failure_count = 0
        total_count = 1000
        for _ in range(total_count):
            w_mlu = torch.empty((100, 100), dtype=torch.float, device="mlu")
            w_mlu.normal_(2, 3)
            mlu_data_ptr_orig = w_mlu.data_ptr()
            # random kernel can't test by comparing with cpu result
            # now only can test function
            self.assertEqual(mlu_data_ptr_orig, w_mlu.data_ptr())
            # use kstest to test the data is normal distribution or not
            res = stats.kstest(w_mlu.cpu().numpy().reshape(-1), "norm", args=(2, 3))
            if res.statistic > 0.1:
                failure_count = failure_count + 1
        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @testinfo()
    def test_normal_out_and_tensor_input(self):
        shape_list = [(2, 7, 34, 66), (34, 50, 6), (20, 8)]
        mean_list = [-10, 0, 3]
        std_list = [1, 5, 2]
        dtype_list = [
            torch.float32,
            torch.half,
            torch.double,
            torch.chalf,
            torch.cfloat,
            torch.cdouble,
        ]
        input_arg_type = ["tensor_tensor", "tensor_float", "float_tensor"]
        out_func_test = [True, False]
        failure_count = 0
        total_count = 0
        product_list = product(
            shape_list, mean_list, std_list, dtype_list, input_arg_type, out_func_test
        )

        # cast complex tensor to real tensor
        def complex_to_real(input):
            if input.is_complex():
                return torch.view_as_real(input)
            else:
                return input

        for shape, mean, std, dtype, arg_type, out_func in product_list:
            org_mean = mean
            org_std = std
            if arg_type == "tensor_tensor":
                mean = torch.tensor(mean, dtype=torch.float32, device="mlu")
                std = torch.tensor(std, dtype=torch.float32, device="mlu")
            elif arg_type == "tensor_float":
                mean = torch.tensor(mean, dtype=torch.float32, device="mlu")
            elif arg_type == "float_tensor":
                std = torch.tensor(std, dtype=torch.float32, device="mlu")

            w_mlu = None
            total_count += 1
            if out_func:
                w_mlu = torch.empty(shape, dtype=dtype, device="mlu")
                torch.normal(mean=mean, std=std, size=shape, out=w_mlu)
            else:
                w_mlu = torch.normal(mean=mean, std=std, size=shape)

            # use kstest to test the data is normal distribution or not
            res = stats.kstest(
                complex_to_real(w_mlu).cpu().numpy().reshape(-1),
                "norm",
                args=(org_mean, org_std),
            )
            if res.statistic > 0.1:
                failure_count += 1

            # fixed shape test
            total_count += 1
            w_mlu = torch.empty(shape, dtype=dtype, device="mlu")
            torch.normal(mean=mean, std=std, size=(100, 100), out=w_mlu)
            # use kstest to test the data is normal distribution or not
            res = stats.kstest(
                complex_to_real(w_mlu).cpu().numpy().reshape(-1),
                "norm",
                args=(org_mean, org_std),
            )
            if res.statistic > 0.1:
                failure_count += 1

            total_count += 1
            out_mlu = torch.normal(mean=mean, std=std, size=shape)
            # use kstest to test the data is normal distribution or not
            res = stats.kstest(
                complex_to_real(out_mlu).cpu().numpy().reshape(-1),
                "norm",
                args=(org_mean, org_std),
            )
            if res.statistic > 0.1:
                failure_count += 1

        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @testinfo()
    def test_normal_channels_last(self):  # pylint: disable=R0201
        layout_list = ["channels_last", "not_dense"]
        shape_list = [(2, 7, 34, 66), (34, 50, 6), (1000, 20, 8)]
        mean_list = [-10, 0, 3]
        std_list = [1, 5, 2]
        dtype_list = [torch.float32, torch.half, torch.double]
        failure_count = 0
        total_count = 0
        product_list = product(layout_list, shape_list, mean_list, std_list, dtype_list)
        for layout, shape, mean, std, dtype in product_list:
            total_count += 1
            w_mlu = torch.empty(shape, dtype=dtype, device="mlu")
            if layout == "channels_last" and len(shape) == 4:
                w_mlu = w_mlu.to(memory_format=torch.channels_last)
            elif layout == "not_dense":
                w_mlu = w_mlu[..., 2]
            # In same small cases, mlu will have inconsistent mean and std,
            # and cpu has a similar situation
            mlu_data_ptr_orig = w_mlu.data_ptr()
            w_mlu.normal_(mean=mean, std=std)
            self.assertEqual(mlu_data_ptr_orig, w_mlu.data_ptr())
            res = stats.kstest(
                w_mlu.cpu().numpy().reshape(-1), "norm", args=(mean, std)
            )
            if res.statistic > 0.1:
                failure_count = failure_count + 1
        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("46GB")
    def test_normal_large(self):
        failure_count = 0
        total_count = 2
        for _ in range(total_count):
            w_mlu = torch.empty((5, 1024, 1024, 1024), dtype=torch.float, device="mlu")
            w_mlu.normal_(2, 3)
            mlu_data_ptr_orig = w_mlu.data_ptr()
            # random kernel can't test by comparing with cpu result
            # now only can test function
            self.assertEqual(mlu_data_ptr_orig, w_mlu.data_ptr())
            # use kstest to test the data is normal distribution or not
            res = stats.kstest(w_mlu.cpu().numpy().reshape(-1), "norm", args=(2, 3))
            if res.statistic > 0.1:
                failure_count = failure_count + 1
        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_normal_bfloat16(self):  # pylint: disable=R0201
        total_count = 1000
        for _ in range(total_count):
            w_mlu = torch.empty((100, 100), dtype=torch.bfloat16, device="mlu")
            w_mlu.normal_(2, 3)
            mlu_data_ptr_orig = w_mlu.data_ptr()
            # random kernel can't test by comparing with cpu result
            # now only can test function
            self.assertEqual(mlu_data_ptr_orig, w_mlu.data_ptr())


if __name__ == "__main__":
    run_tests()
