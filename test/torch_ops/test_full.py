from __future__ import print_function

import sys
import os
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, run_tests, TestCase

logging.basicConfig(level=logging.DEBUG)


class TestFullOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_full(self):
        shape_list = [(10, 3, 512, 224), (2, 3, 4), (0, 3, 4), (2,)]
        value_list = [2.3, 5, 0.59, 0.21, 0]
        dtype_list = [
            (torch.uint8, 0),
            (torch.int8, 0),
            (torch.int16, 0),
            (torch.int32, 0),
            (torch.long, 0),
            (torch.double, 0),
            (torch.float, 0),
            (torch.half, 3e-3),
            (torch.bool, 0),
        ]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in dtype_list:
                out_cpu = torch.full(
                    shape_list[i], value_list[i], dtype=data_type, device="cpu"
                )
                out_mlu_1 = torch.full(
                    shape_list[i], value_list[i], dtype=data_type, device="mlu"
                )
                out_mlu_2 = torch.full(
                    shape_list[i],
                    self.to_mlu(torch.tensor(value_list[i])),
                    dtype=data_type,
                    device="mlu",
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_1.cpu().float(), err, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_2.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_full_bound(self):
        shape_list = [(2, 246625072), (2147483647,)]
        value = 2.3
        data_type = torch.float
        for shape in shape_list:
            out_cpu = torch.full(shape, value, dtype=data_type, device="cpu")
            out_mlu = torch.full(shape, value, dtype=data_type, device="mlu")
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_full_out(self):
        shape_list = [(10, 3, 512, 224), (2, 3, 4), (0, 3, 4), (2,)]
        value_list = [2.3, 5, 0.59, 0.21, 0]
        dtype_list = [
            (torch.uint8, 0),
            (torch.int8, 0),
            (torch.int16, 0),
            (torch.int32, 0),
            (torch.long, 0),
            (torch.double, 0),
            (torch.float, 0),
            (torch.half, 3e-3),
            (torch.bool, 0),
        ]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in dtype_list:
                out = torch.full((2, 3), value_list[i], dtype=data_type, device="cpu")
                out_mlu = out.mlu()
                torch.full(shape_list[i], value_list[i], out=out, dtype=data_type)
                torch.full(shape_list[i], value_list[i], out=out_mlu, dtype=data_type)
                self.assertTensorsEqual(
                    out.float(), out_mlu.cpu().float(), err, use_MSE=True
                )


class TestFullLikeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_full(self):
        shape_list = [(10, 3, 512, 224), (2, 3, 4), (0, 3, 4), (2,)]
        value_list = [2.3, 5, 0.59, 0.21, 0]
        dtype_list = [
            (torch.uint8, 0),
            (torch.int8, 0),
            (torch.int16, 0),
            (torch.int32, 0),
            (torch.long, 0),
            (torch.double, 0),
            (torch.float, 0),
            (torch.half, 3e-3),
            (torch.bool, 0),
        ]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in dtype_list:
                x = torch.randn(shape_list[i])
                out_cpu = torch.full_like(
                    x, value_list[i], dtype=data_type, device="cpu"
                )
                out_mlu_1 = torch.full_like(
                    x, value_list[i], dtype=data_type, device="mlu"
                )
                out_mlu_2 = torch.full_like(
                    x,
                    self.to_mlu(torch.tensor(value_list[i])),
                    dtype=data_type,
                    device="mlu",
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_1.cpu().float(), err, use_MSE=True
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu_2.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_full_bound(self):
        shape_list = [(2, 246625072), (2147483647,)]
        value = 2.3
        data_type = torch.float
        for shape in shape_list:
            x = torch.randn(shape)
            out_cpu = torch.full_like(x, value, dtype=data_type, device="cpu")
            out_mlu = torch.full_like(x, value, dtype=data_type, device="mlu")
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())

    # @unittest.skip("not test")
    @testinfo()
    def test_full_channels_last(self):
        shape_list = [(2, 2, 3, 4), (4, 5, 6, 7)]
        value = 2.3
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.full_like(
                x, value, device="cpu", memory_format=torch.channels_last
            )
            out_mlu = torch.full_like(
                x, value, device="mlu", memory_format=torch.channels_last
            )
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())


class TestNewFullOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_full(self):
        shape_list = [(10, 3, 512, 224), (2, 3, 4), (0, 3, 4), (2,)]
        value_list = [2.3, 5, 0.59, 0.21, 0]
        dtype_list = [
            (torch.uint8, 0),
            (torch.int8, 0),
            (torch.int16, 0),
            (torch.int32, 0),
            (torch.long, 0),
            (torch.double, 0),
            (torch.float, 0),
            (torch.half, 3e-3),
            (torch.bool, 0),
        ]
        new_value = 3.4
        new_shape = (5, 6)
        for i in range(len(shape_list)):  # pylint: disable=C0200
            for data_type, err in dtype_list:
                tmp_out = torch.full(
                    shape_list[i], value_list[i], dtype=data_type, device="cpu"
                )
                tmp_out_mlu = tmp_out.mlu()
                out_cpu = tmp_out.new_full(
                    new_shape, new_value, dtype=data_type, device="cpu"
                )
                out_mlu = tmp_out_mlu.new_full(
                    new_shape, new_value, dtype=data_type, device="mlu"
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_full_bound(self):
        shape = (2, 3)
        value = 0.21
        new_shape_list = [(2, 246625072), (2147483647,)]
        new_value = 2.3
        data_type = torch.float
        for new_shape in new_shape_list:
            tmp_out = torch.full(shape, value, dtype=data_type, device="cpu")
            tmp_out_mlu = tmp_out.mlu()
            out_cpu = tmp_out.new_full(
                new_shape, new_value, dtype=data_type, device="cpu"
            )
            out_mlu = tmp_out_mlu.new_full(
                new_shape, new_value, dtype=data_type, device="mlu"
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )
            self.assertTrue(out_cpu.size() == out_mlu.size())
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTrue(out_mlu.storage_offset() == out_cpu.storage_offset())


if __name__ == "__main__":
    run_tests()
