import sys
import logging
import os
import math
import functools
import itertools
import unittest

# cpu _foreach_norm behave differently when OMP_NUM_THREADS is set
ENV_OLD = os.environ.get("OMP_NUM_THREADS", None)
os.environ.pop("OMP_NUM_THREADS", None)

import torch
from foreach_test_utils import ForeachOpTest, ForeachType

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, run_tests, TestCase

logging.basicConfig(level=logging.DEBUG)


class TestForeachReduceOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_foreach_reduce(self):
        ords = [1, 2, math.inf]
        output_dtype = [None, torch.float, torch.double]
        for ord, dtype in itertools.product(ords, output_dtype):
            kwargs = {"ord": ord, "dtype": dtype}
            test_func = ForeachOpTest(
                torch._foreach_norm, ForeachType.ReduceOp, err=0.003, **kwargs
            )
            test_func(
                self.assertTrue,
                functools.partial(self.assertTensorsEqual, allow_inf=True),
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_foreach_reduce_with_graph(self):
        input = [torch.randn((2, 3), device="mlu") for _ in range(5)]
        graph_mode = ["relaxed", "global"]
        tensor_num = 5
        for mode in graph_mode:
            graph = torch.mlu.MLUGraph()
            with torch.mlu.graph(graph, capture_error_mode=mode):
                output = torch._foreach_norm(input)
            input_real = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
            for i in range(tensor_num):
                input[i].copy_(input_real[i])
            graph.replay()
            input_cpu = [x.cpu() for x in input_real]
            cpu_output = torch._foreach_norm(input_cpu)
            for i in range(tensor_num):
                self.assertTensorsEqual(
                    output[i].cpu(), cpu_output[i], 0.003, use_MSE=True
                )

    def tearDown(self):
        if ENV_OLD:
            os.environ["OMP_NUM_THREADS"] = ENV_OLD
        super().tearDown()


if __name__ == "__main__":
    run_tests()
