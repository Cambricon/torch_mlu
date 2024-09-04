from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import random
from unittest.main import main  # pylint: disable=W0611

import torch
from torch import nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411,W0611

logging.basicConfig(level=logging.DEBUG)


class buildBranchAdd(nn.Module):  # pylint: disable=W0223
    r"""
    graph：
        fc --> split --> permute --> transpose --> \
                    \---------------------------->  add
    """

    def __init__(self, shape):
        super(buildBranchAdd, self).__init__()
        self.shape = len(shape)
        self.s1_dim0_1 = random.randint(3, 25)
        self.s1_dim0_2 = random.randint(3, 25)

        self.s2_dim2 = random.randint(3, 25)

        self.split_dim_0_2 = random.randint(0, 2)
        self.split_dim_0_3 = random.randint(0, 3)
        self.split_dim_0_4 = random.randint(0, 4)
        self.split_dim_0_5 = random.randint(0, 5)

        self.unbind_dim_0_2 = random.randint(0, 2)
        self.unbind_dim_0_3 = random.randint(0, 3)
        self.unbind_dim_0_4 = random.randint(0, 4)
        self.unbind_dim_0_5 = random.randint(0, 5)

        self.select_dim_0_3 = random.randint(0, 3)
        self.select_dim_0_4 = random.randint(0, 4)

        self.narrow_dim_0_2 = random.randint(0, 2)

    def forward(self, x):
        if self.shape == 1:
            dim1 = x.size()[0]
            x = x.unsqueeze(1)
            dim2 = x.size()[1]
            dim0 = self.s1_dim0_1
            x = x.expand(dim0, dim1, dim2)
            x = x.permute(2, 0, 1)
            dim0, dim1, dim2 = dim2, dim0, dim1
            x = x.add(x)
            x = x[:, :, : dim2 - 1]
            x = x.transpose(0, 1)
            dim0, dim2 = self.s1_dim0_2, dim2 - 1
            x = x.expand(dim0, dim1, 1, dim2)
            x = x.squeeze()
            x = x.split(2, self.split_dim_0_2)[0]

        elif self.shape == 2:
            dim0, dim1 = x.size()
            x = x.unsqueeze(1)
            x = x.permute(0, 2, 1)
            dim2 = self.s2_dim2
            x = x.expand(dim2, dim0, dim1, 1)
            x = x.squeeze()
            dim0, dim1, dim2 = dim2, dim0, dim1
            x = x.add(x)
            x = x[:, :, : dim2 - 1]
            x = x.split(2, self.split_dim_0_2)[0]

        elif self.shape == 3:
            dim0, dim1, dim2 = x.size()
            x = x.permute(2, 0, 1)
            x = x.transpose(1, 2)
            x = x.add(x)
            x = x.unsqueeze(2)
            dim0, dim2 = dim2, dim0
            x = x[:, :, :, : dim2 - 1]
            x = x.squeeze()
            x = x.split(2, self.split_dim_0_2)[0]

        elif self.shape == 4:
            x = x.permute(0, 1, 3, 2)
            x = x.transpose(0, 1)
            x = x.add(x)
            dim0, dim1, dim2, dim3 = x.size()
            x = x[:, :, :, : dim3 - 1]
            x = x.split(2, self.split_dim_0_3)[0]
            x = x.unbind(self.unbind_dim_0_3)[0]

        elif self.shape == 5:
            x = x.permute(3, 2, 0, 4, 1)
            x = x.transpose(0, 3)
            x = x.add(x)
            dim0, dim1, dim2, dim3, _ = x.size()
            x = x[:, : dim1 - 1, :, : dim3 - 1, :]
            x = x.split(2, self.split_dim_0_4)[0]
            x = x.unbind(self.unbind_dim_0_4)[0]
            x = x.select(self.select_dim_0_3, 1)

        else:
            x = x.permute(0, 3, 4, 1, 2, 5)
            x = x.transpose(0, 5)
            x = x.add(x)
            dim0, dim1, dim2, dim3, _, dim5 = x.size()
            x = x[:, : dim1 - 1, :, : dim3 - 1, :, : dim5 - 1]
            x = x.split(2, self.split_dim_0_5)[0]
            x = x.unbind(self.unbind_dim_0_5)[0]
            x = x.select(self.select_dim_0_4, 1)
            x = x.unbind(self.unbind_dim_0_2)[0]

        x = x.narrow(self.narrow_dim_0_2, 1, 1)
        x = x.squeeze()
        residual = x
        x = x.transpose(0, 1)
        x = x.permute(1, 0)
        y = x.add(x)
        y = y.chunk(1, 1)[0]
        y = y.squeeze()
        y = y.add(residual)

        return y


class TestResidualNetOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_residual_way(self):
        # print('----Residual-way structure----')
        for d in range(6):
            dim = d + 1
            shape = ()
            for _ in range(1, dim + 1):
                ran_d = random.randint(5, 25)
                shape = shape + (ran_d,)
            data = torch.randn(shape, dtype=torch.float)
            in_cpu = copy.deepcopy(data)
            in_mlu = self.to_mlu(data)
            net_cpu = buildBranchAdd(shape)
            out_cpu = net_cpu(in_cpu)
            out_mlu = net_cpu(in_mlu)
            self.assertTensorsEqual(
                out_cpu, out_mlu.contiguous().cpu().float(), 0.03, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_residual_way_channel_last(self):
        # print('----Residual-way structure----')
        shape = (2, 2, 3, 4)
        data = torch.randn(shape, dtype=torch.float).to(
            memory_format=torch.channels_last
        )
        in_cpu = copy.deepcopy(data)
        in_mlu = self.to_mlu(data)
        net_cpu = buildBranchAdd(shape)
        out_cpu = net_cpu(in_cpu)
        out_mlu = net_cpu(in_mlu)
        self.assertTensorsEqual(
            out_cpu, out_mlu.contiguous().cpu().float(), 0.03, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
