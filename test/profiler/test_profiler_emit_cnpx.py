from __future__ import print_function

import os
import sys
import torch
import unittest
import torch_mlu
from torch.autograd.profiler import emit_cnpx

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase  # pylint: disable=C0411


class TestEmitCnpx(TestCase):
    def test_profiler_emit_cnpx(self):
        # This test is not intended to ensure correctness of cnpx ranges.
        # This test is merely intended to catch if emit_cnpx breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device="mlu")
        with emit_cnpx():
            a.add(1.0)

    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        with torch.autograd.profiler.emit_cnpx(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True, device="mlu")
            y = torch.randn(10, 10, requires_grad=True, device="mlu")
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()


if __name__ == "__main__":
    unittest.main()
