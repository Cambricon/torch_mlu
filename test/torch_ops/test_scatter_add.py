from __future__ import print_function

import sys
import os
import unittest
import logging
import copy

import torch
import torch_mlu
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestScatterAddOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_add_inplace(self):
        shapes = [
            (32, 3, 4, 4, 16, 5),
            (32, 3, 16, 16, 5),
            (32, 3, 16, 16),
            (32, 3, 16),
            (32, 3),
            (32,),
        ]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                x = torch.rand(shape)
                index_cpu = torch.randint(0, shape[dim], shape)
                output_cpu = torch.zeros(shape)
                output_cpu.scatter_add_(dim, index_cpu, x)

                output_mlu = torch.zeros(shape, dtype=torch.float)
                output_mlu = self.to_device(output_mlu)
                index_mlu = self.to_device(index_cpu)
                x_mlu = self.to_device(x)
                output_mlu.scatter_add_(dim, index_mlu, x_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_add(self):
        shapes = [
            (32, 3, 4, 4, 16, 5),
            (2, 3, 4, 4, 4),
            (32, 3, 16, 16),
            (32, 3, 16),
            (32, 3),
            (32,),
        ]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                x = torch.rand(shape)
                index_cpu = torch.randint(0, shape[dim], shape)
                output_cpu = torch.zeros(shape)
                z_cpu = output_cpu.scatter_add(dim, index_cpu, x)

                output_mlu = torch.zeros(shape, dtype=torch.float)
                output_mlu = self.to_device(output_mlu)
                index_mlu = self.to_device(index_cpu)
                x_mlu = self.to_device(x)
                z_mlu = output_mlu.scatter_add(dim, index_mlu, x_mlu)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(z_cpu, z_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_add_channels_last(self):
        shapes = [(100, 512, 2, 5), (100, 512, 2)]
        for shape in shapes:
            for dim in range(-len(shape), len(shape)):
                src_cpu = torch.rand(shape)
                index_cpu = torch.randint(0, shape[dim], shape)
                input_cpu = torch.zeros(shape)
                input_cpu.scatter_add_(dim, index_cpu, src_cpu)

                input_mlu = torch.zeros(shape, dtype=torch.float)
                input_mlu = self.convert_to_channel_last(input_mlu).to("mlu")
                index_mlu = self.convert_to_channel_last(index_cpu).to("mlu")
                src_mlu = self.convert_to_channel_last(src_cpu).to("mlu")
                input_mlu.scatter_add_(dim, index_mlu, src_mlu)
                self.assertTensorsEqual(
                    input_cpu, input_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_zero_element(self):
        a = torch.randn(2, 3, 4)
        a_mlu = a.to("mlu")
        index = torch.randn(
            0,
        ).to(dtype=torch.long)
        index_mlu = index.to("mlu")
        src = torch.randn(2, 3, 4)
        src_mlu = src.to("mlu")
        a.scatter_add_(2, index, src)
        a_mlu.scatter_add_(2, index_mlu, src_mlu)
        self.assertTensorsEqual(a, a_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_add_exception(self):
        a = torch.randn((4, 4), dtype=torch.complex64).mlu()
        index = torch.randint(0, 8, size=(2, 2), dtype=torch.long).mlu()
        src = torch.randn(size=(2, 2), dtype=torch.complex64).mlu()
        ref_msg = f'"MLU scatter_add" not implemented for'
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            a.scatter_add_(0, index, src)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_add_with_stride(self):
        x = torch.randn((108, 23, 8, 14840), dtype=torch.float32)
        index = torch.from_numpy(np.random.randint(0, 8, (108, 23, 8, 1)))
        cpu_out = torch.scatter_add(x, 2, index.expand(108, 23, 8, 14840), x)
        mlu_out = torch.scatter_add(
            x.mlu(), 2, index.mlu().expand(108, 23, 8, 14840), x.mlu()
        )
        self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_add_with_stride_and_high_dims(self):
        expand_shapes = [
            (543, 256, 32),
            (10, 4, 5, 5, 6, 7),
            (1372, 4, 32),
            (1024, 512, 32, 16),
            (8, 8, 4, 8, 4, 8, 2, 8),
        ]
        org_shapes = [
            (1, 256, 1),
            (10, 4, 5, 1, 1, 1),
            (1372, 1, 1),
            (1024, 512, 1, 1),
            (8, 1, 4, 1, 4, 1, 2, 1),
        ]
        for expand_shape, org_shape in zip(expand_shapes, org_shapes):
            idx = torch.randint(0, 2, org_shape)
            idx_mlu = idx.mlu()
            src = torch.randn(expand_shape)
            src_mlu = src.mlu()
            for dim in range(len(expand_shape)):
                cpu_out = torch.scatter_add(src, dim, idx.expand(expand_shape), src)
                mlu_out = torch.scatter_add(
                    src_mlu, dim, idx_mlu.expand(expand_shape), src_mlu
                )
                self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_scatter_add_bfloat16(self):
        shape = (2, 3, 4, 4, 4)
        dim = 2
        x = torch.rand(shape, dtype=torch.bfloat16, requires_grad=True)
        index_cpu = torch.randint(0, shape[dim], shape)
        output_cpu = torch.zeros(shape, dtype=torch.float)
        grad_cpu = torch.rand(shape, dtype=torch.bfloat16)

        z_cpu = output_cpu.scatter_add(dim, index_cpu, x.float())
        z_cpu.backward(grad_cpu.float())

        input_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()

        output_mlu = torch.zeros(shape, dtype=torch.bfloat16)
        output_mlu = self.to_device(output_mlu)
        index_mlu = self.to_device(index_cpu)
        x_mlu = self.to_device(x)
        grad_mlu = self.to_device(grad_cpu)

        z_mlu = output_mlu.scatter_add(dim, index_mlu, x_mlu)
        z_mlu.backward(grad_mlu)

        input_grad_mlu = copy.deepcopy(x.grad)

        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(z_cpu.float(), z_mlu.cpu().float(), 0.003, use_MSE=True)
        self.assertTensorsEqual(
            input_grad_cpu.float(), input_grad_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
