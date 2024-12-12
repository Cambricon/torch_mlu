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


class TestBinaryForeachOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_foreach_binary(self):
        api_list = [
            torch._foreach_add,
            torch._foreach_add_,
            torch._foreach_sub_,
            torch._foreach_sub,
            torch._foreach_mul,
            torch._foreach_mul_,
            torch._foreach_div,
            torch._foreach_div_,
        ]
        foreach_type_list = [
            ForeachType.BinaryOpWithTensor,
            ForeachType.BinaryOpWithScalarList,
            ForeachType.BinaryOpWithScalar,
            ForeachType.BinaryOpWithScalarTensor,
        ]
        for api_func in api_list:
            for foreach_type in foreach_type_list:
                test_func = ForeachOpTest(api_func, foreach_type, err=0.003)
                test_func(self.assertTrue, self.assertTensorsEqual)

    # @unittest.skip("not test")
    @testinfo()
    def test_foreach_binary_with_graph(self):
        api_list = [
            torch._foreach_add,
            torch._foreach_div,
        ]
        tensor_num = 5
        input_left = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        input_right = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        output = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        graph_mode = ["relaxed", "global"]
        for mode in graph_mode:
            for api_func in api_list:
                graph = torch.mlu.MLUGraph()
                with torch.mlu.graph(graph, capture_error_mode=mode):
                    output = api_func(input_left, input_right)
                input_left_real = [
                    torch.randn((2, 3), device="mlu") for _ in range(tensor_num)
                ]
                input_right_real = [
                    torch.randn((2, 3), device="mlu") for _ in range(tensor_num)
                ]
                for i in range(tensor_num):
                    input_left[i].copy_(input_left_real[i])
                    input_right[i].copy_(input_right_real[i])
                    output[i].copy_(input_left_real[i])
                graph.replay()
                input_left_cpu = [item.cpu() for item in input_left_real]
                input_right_cpu = [item.cpu() for item in input_right_real]
                cpu_output = api_func(input_left_cpu, input_right_cpu)
                for i in range(tensor_num):
                    self.assertTensorsEqual(
                        output[i].cpu(), cpu_output[i], 0.003, use_MSE=True
                    )


if __name__ == "__main__":
    unittest.main()
