from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
from torch import nn
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestLinearOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_linear_not_dense(self):
        shape_list = [
            ((10, 100), (50, 25)),
            ((120, 1024), (648, 234)),
            ((33, 150), (32, 12)),
            ((128, 500), (5, 40)),
        ]
        for m1_shape, m2_shape in shape_list:
            m = nn.Linear(m2_shape[0], m2_shape[1])
            input = torch.randn(m1_shape, dtype=torch.float)
            m_mlu = copy.deepcopy(m).to("mlu")
            input_mlu = copy.deepcopy(input).to("mlu")
            input = input[:, 0 : m2_shape[0]]
            input_mlu = input_mlu[:, 0 : m2_shape[0]]
            input.requires_grad = True
            input_mlu.requires_grad = True
            output = m(input)
            output_mlu = m_mlu(input_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu(), 3e-3, use_MSE=True)

            grad = torch.randn(output.shape)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            output.backward(grad)
            output_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                input.grad, input_mlu.grad.cpu(), 3e-3, use_MSE=True
            )


if __name__ == "__main__":
    unittest.main()
