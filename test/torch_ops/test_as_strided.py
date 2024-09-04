from __future__ import print_function

import logging
import sys
import os
import unittest
from itertools import product

import torch
from torch import nn

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestAsStrided(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_as_strided_orig(self):
        shape_list = [(2, 4), (2, 3, 4), (2, 2, 3, 4), (2, 2, 2, 2, 3)]
        size_list = [(1, 2), (2, 3), (1, 2, 3), (2, 2, 2)]
        stride_list = [(1, 2), (1, 1), (2, 1, 2), (1, 1, 1)]
        storage_offset_list = [1, 2]
        for item in product(shape_list, storage_offset_list):
            x = torch.randn(item[0], dtype=torch.float)
            x_mlu = x.to("mlu")
            for stride, size in zip(stride_list, size_list):
                out_cpu = torch.as_strided(x, size, stride, item[1])
                self.assertTrue(x.storage().data_ptr() == out_cpu.storage().data_ptr())
                out_mlu = torch.as_strided(x_mlu, size, stride, item[1])
                # (TODO) mlu tensor not support .storage() operator.
                self.assertTrue(x_mlu.data_ptr() == out_mlu.data_ptr() - 4 * item[1])
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_as_strided_channels_last(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)]
        size_list = [(1, 2), (2, 3), (1, 2, 3), (2, 2, 2)]
        stride_list = [(1, 2), (1, 1), (2, 1, 2), (1, 1, 1)]
        storage_offset_list = [1, 2]
        for item in product(shape_list, storage_offset_list):
            x = torch.randn(item[0], dtype=torch.float).to(
                memory_format=torch.channels_last
            )
            x_mlu = x.to("mlu")
            for stride, size in zip(stride_list, size_list):
                out_cpu = torch.as_strided(x, size, stride, item[1])
                self.assertTrue(x.storage().data_ptr() == out_cpu.storage().data_ptr())
                out_mlu = torch.as_strided(x_mlu, size, stride, item[1])
                # (TODO) mlu tensor not support .storage() operator.
                self.assertTrue(x.stride() == x_mlu.stride())
                self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_as_strided_not_dense(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)]
        size_list = [(1, 2), (2, 3), (1, 2, 3), (2, 2, 2)]
        stride_list = [(1, 2), (1, 1), (2, 1, 2), (1, 1, 1)]
        storage_offset_list = [1, 2]
        for item in product(shape_list, storage_offset_list):
            x = torch.randn(item[0], dtype=torch.float)
            x_cpu = x[:, :, :, 1:3]
            x_mlu = x.to("mlu")[:, :, :, 1:3]
            for stride, size in zip(stride_list, size_list):
                out_cpu = torch.as_strided(x_cpu, size, stride, item[1])
                self.assertTrue(
                    x_cpu.storage().data_ptr() == out_cpu.storage().data_ptr()
                )
                out_mlu = torch.as_strided(x_mlu, size, stride, item[1])
                # (TODO) mlu tensor not support .storage() operator.
                self.assertTrue(x_cpu.stride() == x_mlu.stride())
                self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())
                self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_as_strided_like_expand(self):
        a = torch.rand(4, 3, 2, 121)
        c = torch.rand(4, 3, 2, 121)

        a_mlu = a.to("mlu")
        c_mlu = c.to("mlu")

        a = a.as_strided((4, 3, 2, 121), (363, 121, 0, 1))
        c = c.as_strided((4, 3, 2, 121), (726, 1, 363, 3))

        a_mlu = a_mlu.as_strided((4, 3, 2, 121), (363, 121, 0, 1))
        c_mlu = c_mlu.as_strided((4, 3, 2, 121), (726, 1, 363, 3))

        res_mlu = a_mlu.mul(c_mlu)
        res = a.mul(c)

        self.assertTensorsEqual(res, res_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_as_strided_with_views_op(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = x.view((5, 18, 2, 2, 4))
                x = x.permute(4, 1, 2, 3, 0)
                x = x.as_strided((3, 17, 1, 1, 4), (1, 14, 6, 2, 286), 5)
                x = torch.narrow(x, 4, 3, 1)
                x = x.expand(3, 17, 19, 9, 13)
                return x

        x_cpu = torch.randn((2, 2, 18, 4, 5), dtype=torch.float)
        x_mlu = x_cpu.to("mlu")
        net = Net()
        out_cpu = net(x_cpu)
        out_mlu = net(x_mlu)

        self.assertTrue(x_cpu.stride() == x_mlu.stride())
        self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTrue(out_cpu.size() == out_mlu.size())
        self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_as_strided_exception(self):
        size_list = [(1,), (1, 2), (1, 2, 3), (1, 2), (2, 2)]
        stride_list = [(1, 2), (1,), (2, 3), (1, 2), (-1, 1)]
        storage_offset_list = [1, 0, 2, -1, 1]
        msg_list = [
            r"mismatch in length of strides and shape",
            r"mismatch in length of strides and shape",
            r"mismatch in length of strides and shape",
            r"Tensor: invalid storage offset -1",
            r"as_strided: Negative strides are not supported",
        ]
        x = torch.randn((2, 3, 4), dtype=torch.float)
        x_mlu = x.to("mlu")
        for msg, stride, size, storage_offset in zip(
            msg_list, stride_list, size_list, storage_offset_list
        ):
            with self.assertRaisesRegex(RuntimeError, msg):
                out = torch.as_strided(
                    x_mlu, size, stride, storage_offset
                )  # pylint: disable=W0612

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_as_strided_bfloat16(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                x = x.view((5, 18, 2, 2, 4))
                x = x.permute(4, 1, 2, 3, 0)
                x = x.as_strided((3, 17, 1, 1, 4), (1, 14, 6, 2, 286), 5)
                x = torch.narrow(x, 4, 3, 1)
                x = x.expand(3, 17, 19, 9, 13)
                return x

        x_cpu = torch.randn((2, 2, 18, 4, 5), dtype=torch.bfloat16)
        x_mlu = x_cpu.to("mlu")
        net = Net()
        out_cpu = net(x_cpu)
        out_mlu = net(x_mlu)

        self.assertTrue(x_cpu.stride() == x_mlu.stride())
        self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        self.assertTrue(out_cpu.size() == out_mlu.size())
        self.assertTrue(out_cpu.stride() == out_mlu.stride())
        self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())


if __name__ == "__main__":
    unittest.main()
