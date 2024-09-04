from __future__ import print_function

import sys
import logging
import os
import unittest
import torch
from torch.autograd.function import Function
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestDumpOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_dump(self):
        a = torch.tensor([1.1, 2.2])
        torch.ops.torch_mlu.dump(self.to_mlu(a))
        torch.ops.torch_mlu.dump(self.to_mlu(a.half()))

    # @unittest.skip("not test")
    @testinfo()
    def test_dump_forward_backward(self):
        class MLUDump(Function):
            @staticmethod
            def forward(ctx, a, b):
                # ctx is a context object that can be used to stash information
                # for backward computation
                result = a + b
                dump_success = torch.ops.torch_mlu.dump(result)
                ctx.save_for_backward(a, b, result)
                ctx.dump_success = dump_success
                return result

            @staticmethod
            def backward(ctx, grad_output):
                # We return as many input gradients as there were arguments.
                # Gradients of non-Tensor arguments to forward must be None.
                a, b, result = ctx.saved_tensors
                grad_input = grad_output + a + b * result
                if ctx.dump_success:
                    torch.ops.torch_mlu.dump(grad_input)
                return grad_input, None

        mlu_dump = MLUDump.apply
        a = torch.tensor([1.1, 2.2], requires_grad=True)
        b = torch.tensor([1.1, 2.2], requires_grad=False)
        out = mlu_dump(self.to_mlu(a), self.to_mlu(b))
        grad_output = torch.ones(out.shape, dtype=torch.float) * 2
        out.backward(self.to_mlu(grad_output))


if __name__ == "__main__":
    unittest.main()
