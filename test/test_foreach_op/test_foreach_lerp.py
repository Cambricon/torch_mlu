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


class TestLerpForeachOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_foreach_lerp(self):
        api_list = [
            torch._foreach_lerp,
            torch._foreach_lerp_,
        ]
        foreach_type_list = [
            ForeachType.LerpWithTensor,
            ForeachType.LerpWithScalar,
        ]
        for api_func in api_list:
            for foreach_type in foreach_type_list:
                test_func = ForeachOpTest(api_func, foreach_type, err=0.003)
                test_func(self.assertTrue, self.assertTensorsEqual)

    # TODO(CNNLCORE-21331): Foreach op not support graph now.
    @unittest.skip("not test")
    @testinfo()
    def test_foreach_lerp_with_graph(self):
        tensor_num = 5
        input_left = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        input_right = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        weight = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        output = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        graph_mode = ["relaxed", "global"]
        for mode in graph_mode:
            graph = torch.mlu.MLUGraph()
            with torch.mlu.graph(graph, capture_error_mode=mode):
                output = torch._foreach_lerp(input_left, input_right, weight)
            input_left_real = [
                torch.randn((2, 3), device="mlu") for _ in range(tensor_num)
            ]
            input_right_real = [
                torch.randn((2, 3), device="mlu") for _ in range(tensor_num)
            ]
            weight_real = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
            for i in range(tensor_num):
                input_left[i].copy_(input_left_real[i])
                input_right[i].copy_(input_right_real[i])
                weight[i].copy_(weight_real[i])
                output[i].copy_(input_left_real[i])
            graph.replay()
            input_left_cpu = [item.cpu() for item in input_left_real]
            input_right_cpu = [item.cpu() for item in input_right_real]
            weight_cpu = [item.cpu() for item in weight_real]
            cpu_output = torch._foreach_lerp(
                input_left_cpu, input_right_cpu, weight_cpu
            )
            for i in range(tensor_num):
                self.assertTensorsEqual(
                    output[i].cpu(), cpu_output[i], 0.003, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
