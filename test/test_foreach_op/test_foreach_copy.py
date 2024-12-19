from __future__ import print_function

import sys
import logging
import os
import unittest

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, run_tests, TestCase

logging.basicConfig(level=logging.DEBUG)


class ForeachCopyOpTest(object):
    def __init__(
        self,
        is_mixed_dtype: bool = False,
        err: float = 1e-8,
    ):
        self.func = torch._foreach_copy_
        self.is_mixed_dtype = is_mixed_dtype
        self.device = torch.device("mlu")
        self.input_shapes = [
            (10,),
            (4096000,),
            (3, 4),
            (2, 3, 4),
            (254, 254, 112, 3),
            (40960,),
        ]
        self.err = err
        self.mixed_dtypes = [
            torch.float,
            torch.bfloat16,
            torch.half,
            torch.int32,
            torch.int64,
        ]
        self.dtypes = [
            torch.float,
            torch.bfloat16,
            torch.half,
            torch.int32,
            torch.int64,
        ]

    def generate_inputs(self, dtype):
        inputs = []
        outputs = []
        for idx in range(len(self.input_shapes)):
            input_shape = self.input_shapes[idx]
            if self.is_mixed_dtype:
                input_dtype = self.mixed_dtypes[idx % len(self.mixed_dtypes)]
            else:
                input_dtype = dtype
            inputs.append(
                torch.testing.make_tensor(
                    input_shape, dtype=input_dtype, device=self.device
                )
            )
            outputs.append(
                torch.testing.make_tensor(
                    input_shape, dtype=input_dtype, device=self.device
                )
            )
        return inputs, outputs

    def __call__(self, value_check):
        for dtype in self.dtypes:
            inputs, outputs = self.generate_inputs(dtype)
            torch._foreach_copy_(outputs, inputs)
            for input, output in zip(inputs, outputs):
                value_check(input.float(), output.float(), self.err, use_MSE=True)


class TestForeachCopyOp(TestCase):
    @testinfo()
    def test_foreach_copy(self):
        test_func = ForeachCopyOpTest()
        test_func(self.assertTensorsEqual)

    @testinfo()
    def test_foreach_copy_mixed_dtype(self):
        test_func = ForeachCopyOpTest(is_mixed_dtype=True)
        test_func(self.assertTensorsEqual)

    @testinfo()
    def test_foreach_copy_with_graph(self):
        tensor_num = 5
        inputs = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        outputs = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
        graph_mode = ["relaxed", "global"]
        for mode in graph_mode:
            graph = torch.mlu.MLUGraph()
            with torch.mlu.graph(graph, capture_error_mode=mode):
                output = torch._foreach_copy_(outputs, inputs)
            inputs_real = [torch.randn((2, 3), device="mlu") for _ in range(tensor_num)]
            outputs_real = [
                torch.randn((2, 3), device="mlu") for _ in range(tensor_num)
            ]
            for i in range(tensor_num):
                inputs[i].copy_(inputs_real[i])
                outputs[i].copy_(outputs_real[i])
            graph.replay()
            for i in range(tensor_num):
                self.assertTensorsEqual(outputs[i], inputs[i], 1e-8, use_MSE=True)


if __name__ == "__main__":
    run_tests()
