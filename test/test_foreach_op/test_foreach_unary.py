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


class TestUnaryForeachOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_foreach_unary(self):
        test_func = ForeachOpTest(torch._foreach_zero_, ForeachType.UnaryOp, err=0.0)
        test_func(self.assertTrue, self.assertTensorsEqual)

    # TODO(CNNLCORE-21331): Foreach op not support graph now.
    @unittest.skip("not test")
    @testinfo()
    def test_foreach_unary_with_graph(self):
        input = [torch.randn((2, 3), device="mlu") for _ in range(5)]
        graph_mode = ["relaxed", "global"]
        tensor_num = 5
        for mode in graph_mode:
            graph = torch.mlu.MLUGraph()
            with torch.mlu.graph(graph, capture_error_mode=mode):
                torch._foreach_zero_(input)
            input_real = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
            for i in range(tensor_num):
                input[i].copy_(input_real[i])
            graph.replay()
            for i in range(tensor_num):
                self.assertTrue(torch.sum(input[i]).item() == 0)


if __name__ == "__main__":
    unittest.main()
