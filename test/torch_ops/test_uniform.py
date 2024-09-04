from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from scipy import stats

import torch
from torch import nn

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
torch.manual_seed(2)


class TestUniformOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_orig(self):
        data_types = [
            torch.float,
            torch.half,
            torch.double,
            torch.complex128,
            torch.complex64,
        ]
        layout_list = ["contiguous", "channels_last", "not_dense"]
        for layout in layout_list:
            for data_type in data_types:
                # cpu is not support torch.complex32
                w = torch.randn(3, 4, 3, 3, dtype=data_type)
                if layout == "channels_last":
                    w = w.to(memory_format=torch.channels_last)
                elif layout == "not_dense":
                    w = w[..., 2]
                w_copy = copy.deepcopy(w)
                w_mlu = w_copy.to("mlu")
                # random kernel can't test by comparing with cpu result
                # now only can test function
                m = nn.init.uniform_(w)
                mlu_data_ptr_orig = w_mlu.data_ptr()
                m_mlu = nn.init.uniform_(w_mlu)
                self.assertLessEqual(
                    torch.max(m.to(torch.float)), 1, "cpu uniform are not right"
                )
                # MLU cast is not support complex.
                self.assertLessEqual(
                    torch.max(m_mlu.cpu().float()), 1, "mlu uniform are not right"
                )
                self.assertLessEqual(
                    0, torch.min(m.to(torch.float)), "cpu uniform are not right"
                )
                self.assertLessEqual(
                    0, torch.min(m_mlu.cpu().float()), "mlu uniform are not right"
                )
                self.assertEqual(mlu_data_ptr_orig, m_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_range(self):
        # uniform_ can't run on int, short, int8 and so on, so don't test dtypes
        shape_list = [(11), (24, 57), (15, 34, 66), (112, 23, 145, 156)]
        range_list = [(0.1, 0.5), (0.2, 1), (0.1, 2.3), (-24.24, 42.42)]
        for shape in shape_list:
            for scope in range_list:
                w = torch.randn(shape, dtype=torch.float)
                w_copy = copy.deepcopy(w)
                w_mlu = w_copy.to("mlu")
                mlu_data_ptr_orig = w_mlu.data_ptr()
                m = nn.init.uniform_(w, a=scope[0], b=scope[1])
                m_mlu = nn.init.uniform_(w_mlu, a=scope[0], b=scope[1])
                self.assertLessEqual(
                    torch.max(m), scope[1], "cpu uniform are not right"
                )
                self.assertLessEqual(
                    torch.max(m_mlu.cpu()), scope[1], "mlu uniform are not right"
                )
                self.assertLessEqual(
                    scope[0], torch.min(m), "cpu uniform are not right"
                )
                self.assertLessEqual(
                    scope[0], torch.min(m_mlu.cpu()), "mlu uniform are not right"
                )
                self.assertEqual(mlu_data_ptr_orig, m_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_zero_elements(self):
        x = torch.randn((2, 0, 3), dtype=torch.float)
        x_mlu = copy.deepcopy(x).to("mlu")
        # return empty tensor
        out = x.uniform_(0.1, 0.5)
        out_mlu = x_mlu.uniform_(0.1, 0.5)
        self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_stats(self):
        shape_list = [(15, 34, 66), (12, 23, 15, 56)]
        range_list = [(0, 1), (0.2, 1.2)]
        failure_count = 0
        total_count = 0
        TEST_NUM = 20  # set test_num as 20
        for _ in range(TEST_NUM):
            for shape in shape_list:
                for scope in range_list:
                    total_count += 1
                    w = torch.randn(shape, dtype=torch.float)
                    w_mlu = w.to("mlu")
                    m_mlu = nn.init.uniform_(w_mlu, a=scope[0], b=scope[1])
                    # use kstest to test the data is uniform distribution or not
                    p_value = stats.kstest(
                        m_mlu.cpu().numpy().reshape(-1),
                        "uniform",
                        args=scope,
                        N=w.numel(),
                        alternative="less",
                    )
                    if p_value[1] <= 0.005:
                        failure_count = failure_count + 1
        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_channels_last(self):
        range_list = [
            (0, 1),
        ]
        layout_list = ["channels_last", "not_dense"]
        failure_count = 0
        total_count = 0
        TEST_NUM = 20  # set test_num as 20
        for _ in range(TEST_NUM):
            for layout in layout_list:
                for scope in range_list:
                    total_count += 1
                    w = torch.randn((2, 7, 34, 66), dtype=torch.float)
                    if layout == "channels_last":
                        w = w.to(memory_format=torch.channels_last)
                    elif layout == "not_dense":
                        w = w[..., 2]
                    w_mlu = w.to("mlu")
                    m_mlu = nn.init.uniform_(w_mlu, a=scope[0], b=scope[1])
                    # use kstest to test the data is uniform distribution or not
                    p_value = stats.kstest(
                        m_mlu.cpu().numpy().reshape(-1),
                        "uniform",
                        args=scope,
                        N=w.numel(),
                        alternative="less",
                    )
                    if p_value[1] <= 0.005:
                        failure_count = failure_count + 1
        self.assertTrue(failure_count < total_count * 0.1)

    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_int(self):
        range_list = [
            (0, 10000),
        ]
        layout_list = ["contiguous", "channels_last", "not_dense"]
        failure_count = 0
        total_count = 0
        TEST_NUM = 20  # set test_num as 20
        for _ in range(TEST_NUM):
            for layout in layout_list:
                for scope in range_list:
                    total_count += 1
                    w = torch.randn((2, 7, 34, 66), dtype=torch.float)
                    if layout == "channels_last":
                        w = w.to(memory_format=torch.channels_last)
                    elif layout == "not_dense":
                        w = w[..., 2]
                    w_mlu = w
                    m_mlu = w_mlu.random_(scope[0], scope[1])
                    # use kstest to test the data is uniform distribution or not
                    p_value = stats.kstest(
                        m_mlu.cpu().numpy().reshape(-1),
                        "randint",
                        args=scope,
                        N=w.numel(),
                        alternative="less",
                    )
                    if p_value[1] <= 0.005:
                        failure_count = failure_count + 1
        self.assertTrue(failure_count < total_count * 0.1)

    @unittest.skip("not test")
    @testinfo()
    def test_uniform_int_from_to(self):
        type_list = [torch.float, torch.half, torch.int]
        for t in type_list:
            a = torch.randn(10, dtype=torch.float).to(t).to("mlu")
            # from=0, to=10
            a = a.random_(to=10)
            # from=-10, to is the max value of type
            a = a.random_(-10, None)
            # from=-10, to=10
            a = a.random_(-10, 10)

    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_int_empty(self):
        a = torch.randn((0, 1)).to("mlu")
        a = a.random_(-10, 10)

    # @unittest.skip("not test")
    @testinfo()
    def test_uniform_exceptions(self):
        shape = (15, 34, 66)
        range = (2.2, 1.2)
        float_min = torch.finfo(torch.float).min
        float_max = torch.finfo(torch.float).max
        double_min = torch.finfo(torch.double).min
        double_max = torch.finfo(torch.double).max
        w = torch.randn(shape, dtype=torch.float).mlu()
        with self.assertRaises(RuntimeError) as info:
            w.uniform_(range[0], range[1])
        msg = (
            "uniform_ expects to return a [from, to) range, "
            "but found from=2.2 > to=1.2"
        )
        self.assertEqual(info.exception.args[0], msg)

        with self.assertRaises(RuntimeError) as info:
            w.uniform_(range[0], double_max)
        msg = "to is out of bounds for float"
        self.assertEqual(info.exception.args[0], msg)

        with self.assertRaises(RuntimeError) as info:
            w.uniform_(double_min, 1)
        msg = "from is out of bounds for float"
        self.assertEqual(info.exception.args[0], msg)

        with self.assertRaises(RuntimeError) as info:
            w.uniform_(float_min, float_max)
        msg = (
            "uniform_ expects to-from <= std::numeric_limits<Float>::max(), "
            "but found to=3.40282e+38 and from=-3.40282e+38 which result in "
            "to-from to exceed the limit"
        )
        self.assertEqual(info.exception.args[0], msg)

        size, dtype = 2000, torch.int32
        t = torch.empty(size, dtype=dtype, device="mlu")
        with self.assertRaises(RuntimeError) as info:
            t.uniform_(1, 10)
        msg = "\"check_uniform_bounds\" not implemented for 'Int'"
        self.assertEqual(info.exception.args[0], msg)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("46GB")
    def test_uniform_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        range_list = [(0, 1)]
        failure_count = 0
        total_count = 0
        TEST_NUM = 2  # set test_num as 20
        for _ in range(TEST_NUM):
            for shape in shape_list:
                for scope in range_list:
                    total_count += 1
                    w = torch.randn(shape, dtype=torch.float)
                    w_mlu = w.to("mlu")
                    m_mlu = nn.init.uniform_(w_mlu, a=scope[0], b=scope[1])
                    # use kstest to test the data is uniform distribution or not
                    p_value = stats.kstest(
                        m_mlu.cpu().numpy().reshape(-1),
                        "uniform",
                        args=scope,
                        N=w.numel(),
                        alternative="less",
                    )
                    if p_value[1] <= 0.005:
                        failure_count = failure_count + 1
        self.assertTrue(failure_count < total_count * 0.1)


if __name__ == "__main__":
    run_tests()
