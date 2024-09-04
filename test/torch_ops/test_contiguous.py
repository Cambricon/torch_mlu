from __future__ import print_function

import sys
import os
import copy
import unittest
import logging

import torch
import torch.autograd

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_slice(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            shape_list = [(160, 4), (897, 1, 8), (113, 1, 1)]
            for shape in shape_list:
                input = torch.rand(shape, dtype=torch.float)
                out_cpu = input[:, 0::4] / 1.0
                input_mlu = self.to_mlu_dtype(input, data_type)
                out_mlu = input_mlu[:, 0::4] / 1.0

                # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_slice_None(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            shape_list = [(160, 4), (897, 1, 8), (113, 1, 1)]
            for shape in shape_list:
                x = torch.rand(shape, dtype=torch.float)
                out_cpu = x[:, None, 2:]
                x_mlu = self.to_mlu_dtype(x, data_type)
                out_mlu = x_mlu[:, None, 2:]

                # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_expand(self):
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for data_type, err in dtype_list:
            shape_list = [((9, 4), (160, 4)), ((33, 8), (47, 8)), ((13, 1), (59, 1))]
            for shape1, shape2 in shape_list:
                box1 = torch.rand(shape1, dtype=torch.float)
                box2 = torch.rand(shape2, dtype=torch.float)

                N = box1.size(0)
                M = box2.size(0)

                be1 = box1.unsqueeze(1).expand(-1, M, -1)
                be2 = box2.unsqueeze(0).expand(N, -1, -1)

                out_cpu = torch.max(be1[:, :, :2], be2[:, :, :2])

                box1_mlu = self.to_mlu_dtype(box1, data_type)
                box2_mlu = self.to_mlu_dtype(box2, data_type)

                be1_mlu = box1_mlu.unsqueeze(1).expand(-1, M, -1)
                be2_mlu = box2_mlu.unsqueeze(0).expand(N, -1, -1)

                out_mlu = torch.max(be1_mlu[:, :, :2], be2_mlu[:, :, :2])

                # float type precision : 0.003
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_not_dense(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x[:, :, :, 1:3]
            x_mlu = x.to("mlu")[:, :, :, 1:3]
            out_cpu = x_cpu.contiguous()
            out_mlu = x_mlu.contiguous()
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_channels_last(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x.to(memory_format=torch.channels_last)
            x_mlu = x.to("mlu").to(memory_format=torch.channels_last)
            out_cpu = x_cpu.contiguous()
            out_mlu = x_mlu.contiguous()
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_permute(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7), (3, 3, 3, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x.to(memory_format=torch.channels_last).permute(1, 0, 2, 3)
            x_mlu = (
                x.to("mlu").to(memory_format=torch.channels_last).permute(1, 0, 2, 3)
            )
            out_cpu = x_cpu.contiguous()
            out_mlu = x_mlu.contiguous()
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_like_permute(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7), (3, 3, 3, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_cpu = x.to(memory_format=torch.channels_last).permute(1, 0, 2, 3)
            x_mlu = (
                x.to("mlu")
                .to(memory_format=torch.channels_last)
                .as_strided(x_cpu.size(), x_cpu.stride())
            )
            out_cpu = x_cpu.contiguous()
            out_mlu = x_mlu.contiguous()
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_like_expand(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7), (3, 3, 3, 3)]
        expand_value = [(3), (5), (7)]
        for shape in shape_list:
            for value in expand_value:
                x = torch.randn(shape, dtype=torch.float)
                x_cpu = x.unsqueeze(2).expand(-1, -1, value, -1, -1)
                x_mlu = x.to("mlu").as_strided(x_cpu.size(), x_cpu.stride())
                out_cpu = x_cpu.contiguous()
                out_mlu = x_mlu.contiguous()
                self.assertTrue(x_cpu.stride() == x_mlu.stride())
                self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())
                self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_contiguous_like_permute_and_expand(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7), (3, 3, 3, 3)]
        expand_value = [(3), (5), (7)]
        for shape in shape_list:
            for value in expand_value:
                x = (
                    torch.randn(shape, dtype=torch.float)
                    .to(memory_format=torch.channels_last)
                    .permute(1, 0, 2, 3)
                )
                x_cpu = x.unsqueeze(2).expand(-1, -1, value, -1, -1)
                x_mlu = x.to("mlu").as_strided(x_cpu.size(), x_cpu.stride())
                out_cpu = x_cpu.contiguous()
                out_mlu = x_mlu.contiguous()
                self.assertTrue(x_cpu.stride() == x_mlu.stride())
                self.assertTrue(x_mlu.storage_offset() == x_cpu.storage_offset())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())
                self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())


if __name__ == "__main__":
    unittest.main()
