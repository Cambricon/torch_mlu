from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_atan2(self):
        shape_list = [()]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            other = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.atan2(x, other)
            out_mlu = torch.atan2(x, other.mlu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_atan2_type(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4)]
        type_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.short,
            torch.bool,
        ]
        for shape in shape_list:
            for type in type_list:
                x_cpu = torch.randn(shape).to(type)
                other_cpu = torch.randn(shape).to(type)
                x_mlu = self.to_mlu(x_cpu)
                other_mlu = self.to_mlu(other_cpu)
                if type == torch.half:
                    x_cpu = x_cpu.float()
                    other_cpu = other_cpu.float()
                out_cpu = torch.atan2(x_cpu, other_cpu)
                out_mlu = torch.atan2(x_mlu, other_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_atan2_inplace(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (0, 6),
            (),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            other = torch.randn(shape, dtype=torch.float)
            other_mlu = other.mlu()
            x_mlu = self.to_mlu(x)
            x_ptr = x_mlu.data_ptr()
            x.atan2_(other)
            x_mlu.atan2_(other_mlu)
            self.assertEqual(x_ptr, x_mlu.data_ptr())
            self.assertTensorsEqual(x.float(), x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_atan2_out(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (0, 6),
            (),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            other = torch.randn(shape, dtype=torch.float)
            other_mlu = other.mlu()
            out_cpu = torch.randn(1, dtype=torch.float)
            x_mlu = x.mlu()
            out_mlu = torch.randn(1, dtype=torch.float).mlu()
            torch.atan2(x, other, out=out_cpu)
            torch.atan2(x_mlu, other_mlu, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_atan2_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4), (254, 254, 112, 1, 1, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            other = torch.randn(shape, dtype=torch.float)
            other = other[..., : int(shape[-1] / 2)]
            other_mlu = other.mlu()[..., : int(shape[-1] / 2)]
            x_cpu = x[..., : int(shape[-1] / 2)]
            x_mlu = x.mlu()[..., : int(shape[-1] / 2)]
            out_cpu = torch.atan2(x_cpu, other)
            out_mlu = torch.atan2(x_mlu, other_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_atan2_channels_last(self):
        shape_list = [(512, 2, 2, 4), (3, 2, 3, 4), (254, 254, 1, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x.to(memory_format=torch.channels_last)
            x_mlu = x.mlu().to(memory_format=torch.channels_last)
            other = torch.randn(shape, dtype=torch.float)
            other_mlu = other.mlu().to(memory_format=torch.channels_last)
            out_cpu = torch.atan2(x_cpu, other.to(memory_format=torch.channels_last))
            out_mlu = torch.atan2(x_mlu, other_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertEqual(
                out_mlu.is_contiguous(memory_format=torch.preserve_format),
                out_cpu.is_contiguous(memory_format=torch.preserve_format),
            )
            self.assertEqual(
                out_mlu.is_contiguous(memory_format=torch.channels_last),
                out_cpu.is_contiguous(memory_format=torch.channels_last),
            )
            self.assertEqual(
                out_mlu.is_contiguous(memory_format=torch.contiguous_format),
                out_cpu.is_contiguous(memory_format=torch.contiguous_format),
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_atan2_inplace_not_dense(self):
        shape_list = [(512, 1024, 2, 2, 4), (2, 3, 4), (254, 254, 112, 1, 1, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x[..., : int(shape[-1] / 2)]
            x_mlu = x.mlu()[..., : int(shape[-1] / 2)]
            other = torch.randn(shape, dtype=torch.float)
            other_mlu = other.mlu()[..., : int(shape[-1] / 2)]
            other = other[..., : int(shape[-1] / 2)]
            x_ptr = x_mlu.data_ptr()
            x_cpu.atan2_(other)
            x_mlu.atan2_(other_mlu)
            self.assertEqual(x_ptr, x_mlu.data_ptr())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_atan2_inplace_channel_last(self):
        shape_list = [(512, 2, 2, 4), (3, 2, 3, 4), (254, 254, 1, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x.to(memory_format=torch.channels_last)
            x_mlu = x.mlu().to(memory_format=torch.channels_last)
            x_ptr = x_mlu.data_ptr()
            other = torch.randn(shape, dtype=torch.float)
            other_mlu = other.mlu().to(memory_format=torch.channels_last)
            other.to(memory_format=torch.channels_last)
            x_cpu.atan2_(other)
            x_mlu.atan2_(other_mlu)
            self.assertEqual(x_ptr, x_mlu.data_ptr())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu(), 0.003, use_MSE=True)
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.preserve_format),
                x_cpu.is_contiguous(memory_format=torch.preserve_format),
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.channels_last),
                x_cpu.is_contiguous(memory_format=torch.channels_last),
            )
            self.assertEqual(
                x_mlu.is_contiguous(memory_format=torch.contiguous_format),
                x_cpu.is_contiguous(memory_format=torch.contiguous_format),
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_atan2_bfloat16(self):
        shape = (3, 2, 3, 4)
        x = torch.randn(shape, dtype=torch.bfloat16)
        other = torch.randn(shape, dtype=torch.bfloat16)
        out_cpu = torch.atan2(x, other)
        out_mlu = torch.atan2(x.mlu(), other.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
