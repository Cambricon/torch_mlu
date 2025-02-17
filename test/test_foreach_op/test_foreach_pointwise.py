from __future__ import print_function

import sys
import logging
import os
import unittest

import torch
import torch_mlu
from foreach_test_utils import ForeachOpTest, ForeachType

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, run_tests, TestCase

logging.basicConfig(level=logging.DEBUG)


class TestForeachPointWiseOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_foreach_pointwise(self):
        api_list = [
            torch._foreach_addcmul,
            torch._foreach_addcmul_,
            torch._foreach_addcdiv,
            torch._foreach_addcdiv_,
        ]
        foreach_type_list = [
            ForeachType.PointWiseOpWithScalar,
            ForeachType.PointWiseOpWithScalarList,
            ForeachType.PointWiseOpWithScalarTensor,
        ]
        for api_func in api_list:
            for foreach_type in foreach_type_list:
                test_func = ForeachOpTest(api_func, foreach_type, err=0.003)
                test_func(self.assertTrue, self.assertTensorsEqual)


if __name__ == "__main__":
    run_tests()
