from __future__ import print_function

import sys
import logging
import os
import unittest
import copy
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0411,C0413

logging.basicConfig(level=logging.DEBUG)


class TestReplicationPadOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_replication_pad2d(self):
        shape_list = [(2, 3, 4, 5), (32, 3, 224, 224), (2, 3, 4)]
        pad_list = [(1, 1, 2, 3), (0, 2, 1, 3), 1]
        type_list = [(torch.double, 0.0), (torch.float, 0.0), (torch.half, 0.003)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, type_list, func_list]
        for shape, pad, (dtype, err), func in product(*loop_list):
            m = torch.nn.ReplicationPad2d(pad)
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = m(func(x))
            out_mlu = m(func(x_mlu.to(dtype).mlu()))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).mlu()
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_replication_pad2d_exception(self):
        x = torch.randn(2, 3, 4, 5).mlu()
        ref_msg = "padding size is expected to be 4"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch._C._nn.replication_pad2d(x, (0, 1))

    # @unittest.skip("not test")
    @testinfo()
    def test_replication_pad1d(self):
        shape_list = [
            (
                2,
                3,
                4,
            ),
            (32, 3, 224),
            (3, 4),
        ]
        pad_list = [(1, 1), (0, 2), 1]
        type_list = [(torch.double, 0.0), (torch.float, 0.0), (torch.half, 0.003)]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, type_list, func_list]
        for shape, pad, (dtype, err), func in product(*loop_list):
            m = torch.nn.ReplicationPad1d(pad)
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = m(func(x))
            out_mlu = m(func(x_mlu.to(dtype).mlu()))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).mlu()
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_replication_pad1d_exception(self):
        x = torch.randn(3, 4, 5).mlu()
        ref_msg = "padding size is expected to be 2"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch._C._nn.replication_pad1d(x, (0, 1, 2, 3))

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_replication_pad2d_bfloat16(self):
        shape_list = [
            (2, 3, 4, 5),
        ]
        pad_list = [
            (1, 1, 2, 3),
        ]
        type_list = [
            (torch.bfloat16, 0.003),
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, type_list, func_list]
        for shape, pad, (dtype, err), func in product(*loop_list):
            m = torch.nn.ReplicationPad2d(pad)
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = copy.deepcopy(x)
            x.requires_grad = True
            x_mlu.requires_grad = True
            out_cpu = m(func(x))
            out_mlu = m(func(x_mlu.to(dtype).mlu()))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), err, use_MSE=True)

            grad = torch.randn(out_cpu.shape)
            grad_mlu = copy.deepcopy(grad).mlu()
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(x.grad, x_mlu.grad.float(), err, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
