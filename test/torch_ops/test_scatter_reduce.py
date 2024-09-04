import sys
import os
import unittest
import logging
import copy

import torch
import torch_mlu
from itertools import product
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestScatterReduceOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce_inplace(self):
        reduce_list = ["sum", "amax", "amin"]
        shapes = [
            (32, 3, 4, 4, 16, 5),
            (32, 3, 16, 16, 5),
            (32, 3, 16, 16),
            (32, 3, 16),
            (32, 3),
            (32,),
        ]
        for shape, reduce_mode in product(shapes, reduce_list):
            for dim in range(-len(shape), len(shape)):
                x = torch.rand(shape)
                index_cpu = torch.randint(0, shape[dim], shape)
                output_cpu = torch.zeros(shape)
                output_cpu.scatter_reduce_(dim, index_cpu, x, reduce=reduce_mode)

                output_mlu = torch.zeros(shape, dtype=torch.float)
                output_mlu = self.to_device(output_mlu)
                index_mlu = self.to_device(index_cpu)
                x_mlu = self.to_device(x)
                output_mlu.scatter_reduce_(dim, index_mlu, x_mlu, reduce=reduce_mode)
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce(self):
        reduce_list = ["sum", "amax", "amin"]
        shapes = [
            (32, 3, 4, 4, 16, 5),
            (2, 3, 4, 4, 4),
            (32, 3, 16, 16),
            (32, 3, 16),
            (32, 3),
            (32,),
        ]
        for shape, reduce_mode in product(shapes, reduce_list):
            for dim in range(-len(shape), len(shape)):
                x = torch.rand(shape)
                index_cpu = torch.randint(0, shape[dim], shape)
                output_cpu = torch.zeros(shape)
                z_cpu = output_cpu.scatter_reduce(dim, index_cpu, x, reduce=reduce_mode)

                output_mlu = torch.zeros(shape, dtype=torch.float)
                output_mlu = self.to_device(output_mlu)
                index_mlu = self.to_device(index_cpu)
                x_mlu = self.to_device(x)
                z_mlu = output_mlu.scatter_reduce(
                    dim, index_mlu, x_mlu, reduce=reduce_mode
                )
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(z_cpu, z_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce_channels_last(self):
        reduce_list = ["sum", "amax", "amin"]
        shapes = [(100, 512, 2, 5), (100, 512, 2)]
        for shape, reduce_mode in product(shapes, reduce_list):
            for dim in range(-len(shape), len(shape)):
                src_cpu = torch.rand(shape)
                index_cpu = torch.randint(0, shape[dim], shape)
                input_cpu = torch.zeros(shape)
                input_cpu.scatter_reduce_(dim, index_cpu, src_cpu, reduce=reduce_mode)

                input_mlu = torch.zeros(shape, dtype=torch.float)
                input_mlu = self.convert_to_channel_last(input_mlu).to("mlu")
                index_mlu = self.convert_to_channel_last(index_cpu).to("mlu")
                src_mlu = self.convert_to_channel_last(src_cpu).to("mlu")
                input_mlu.scatter_reduce_(dim, index_mlu, src_mlu, reduce=reduce_mode)
                self.assertTensorsEqual(
                    input_cpu, input_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce_zero_element(self):
        reduce_list = ["sum", "amax", "amin"]
        for reduce_mode in reduce_list:
            a = torch.randn(2, 3, 4)
            a_mlu = a.to("mlu")
            index = torch.randn(
                0,
            ).to(dtype=torch.long)
            index_mlu = index.to("mlu")
            src = torch.randn(2, 3, 4)
            src_mlu = src.to("mlu")
            a.scatter_reduce_(2, index, src, reduce=reduce_mode)
            a_mlu.scatter_reduce_(2, index_mlu, src_mlu, reduce=reduce_mode)
            self.assertTensorsEqual(a, a_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce_dtype_exception(self):
        reduce_list = ["sum", "amax", "amin"]
        for reduce_mode in reduce_list:
            a = torch.randn((4, 4), dtype=torch.complex64).mlu()
            index = torch.randint(0, 8, size=(2, 2), dtype=torch.long).mlu()
            src = torch.randn(size=(2, 2), dtype=torch.complex64).mlu()
            ref_msg = f'"MLU scatter_reduce" not implemented for'
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                a.scatter_reduce_(0, index, src, reduce=reduce_mode)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce_mode_exception(self):
        reduce_list = ["prod", "mean"]
        for reduce_mode in reduce_list:
            a = torch.randn((4, 4)).mlu()
            index = torch.randint(0, 8, size=(2, 2)).mlu()
            src = torch.randn(size=(2, 2)).mlu()
            ref_msg = f"MLU scatter reduce of {reduce_mode} is not supported"
            with self.assertRaisesRegex(RuntimeError, ref_msg):
                a.scatter_reduce_(0, index, src, reduce=reduce_mode)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce_with_stride(self):
        reduce_list = ["sum", "amax", "amin"]
        for reduce_mode in reduce_list:
            x = torch.randn((108, 23, 8, 14840), dtype=torch.float32)
            index = torch.from_numpy(np.random.randint(0, 8, (108, 23, 8, 1)))
            cpu_out = torch.scatter_reduce(
                x, 2, index.expand(108, 23, 8, 14840), x, reduce=reduce_mode
            )
            mlu_out = torch.scatter_reduce(
                x.mlu(),
                2,
                index.mlu().expand(108, 23, 8, 14840),
                x.mlu(),
                reduce=reduce_mode,
            )
            self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_scatter_reduce_with_stride_and_high_dims(self):
        reduce_list = ["sum", "amax", "amin"]
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
            for reduce_mode in reduce_list:
                idx = torch.randint(0, 2, org_shape)
                idx_mlu = idx.mlu()
                src = torch.randn(expand_shape)
                src_mlu = src.mlu()
                for dim in range(len(expand_shape)):
                    cpu_out = torch.scatter_reduce(
                        src, dim, idx.expand(expand_shape), src, reduce=reduce_mode
                    )
                    mlu_out = torch.scatter_reduce(
                        src_mlu,
                        dim,
                        idx_mlu.expand(expand_shape),
                        src_mlu,
                        reduce=reduce_mode,
                    )
                    self.assertTensorsEqual(cpu_out, mlu_out.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_scatter_reduce_bfloat16(self):
        shape = (2, 3, 4, 4, 4)
        dim = 2
        reduce_list = ["sum", "amax", "amin"]
        for reduce_mode in reduce_list:
            x = torch.rand(shape, dtype=torch.bfloat16, requires_grad=True)
            index_cpu = torch.randint(0, shape[dim], shape)
            output_cpu = torch.zeros(shape, dtype=torch.float)
            grad_cpu = torch.rand(shape, dtype=torch.bfloat16)

            z_cpu = output_cpu.scatter_reduce(
                dim, index_cpu, x.float(), reduce=reduce_mode
            )
            z_cpu.backward(grad_cpu.float())

            input_grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()

            output_mlu = torch.zeros(shape, dtype=torch.bfloat16)
            output_mlu = self.to_device(output_mlu)
            index_mlu = self.to_device(index_cpu)
            x_mlu = self.to_device(x)
            grad_mlu = self.to_device(grad_cpu)

            z_mlu = output_mlu.scatter_reduce(dim, index_mlu, x_mlu, reduce=reduce_mode)
            z_mlu.backward(grad_mlu)

            input_grad_mlu = copy.deepcopy(x.grad)

            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                z_cpu.float(), z_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                input_grad_cpu.float(),
                input_grad_mlu.cpu().float(),
                0.003,
                use_MSE=True,
            )


if __name__ == "__main__":
    unittest.main()
