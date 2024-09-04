from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
import copy

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(1234)


class TestAllOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_all_dim(self):
        shape_list = [
            (10, 11, 20),
            (1111,),
            (2, 3, 4, 8, 10),
            (34, 56, 78, 90),
            (),
            (0, 6),
        ]
        dim_list = [-2, 0, -1, 3, 0, 1]
        dtype_list = [
            torch.bool,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float,
            torch.half,
            torch.double,
        ]
        keep_type = [True, False]
        for i, list_ in enumerate(shape_list):
            for dtype in dtype_list:
                for keep in keep_type:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    x_mlu = x_cpu.to("mlu")
                    out_cpu = x_cpu.all(dim=dim_list[i], keepdim=keep)
                    out_mlu = x_mlu.all(dim=dim_list[i], keepdim=keep)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                    )
                    self.assertTrue(
                        out_mlu.dtype == out_cpu.dtype, "all out dtype is not right"
                    )
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

        # not contiguous
        shape_list_nc = [(10, 11, 20), (2, 3, 4, 8, 10), (34, 56, 78, 90), (0, 6)]
        dim_list_nc = [-2, -1, 3, 1]
        dtype_list_nc = [
            torch.bool,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float,
            torch.half,
            torch.double,
        ]
        keep_type_nc = [True, False]
        for i, list_ in enumerate(shape_list_nc):
            for dtype in dtype_list_nc:
                for keep in keep_type_nc:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                    if x_cpu.dim() == 4:
                        x_cpu = x_cpu.to(memory_format=torch.channels_last)
                    x_mlu = x_cpu.to("mlu")
                    out_cpu = x_cpu[:, 1:].all(dim=dim_list_nc[i], keepdim=keep)
                    out_mlu = x_mlu[:, 1:].all(dim=dim_list_nc[i], keepdim=keep)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                    )
                    self.assertTrue(
                        out_mlu.dtype == out_cpu.dtype, "all out dtype is not right"
                    )
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_all(self):
        shape_list = [
            (10, 11, 20),
            (1111,),
            (2, 3, 4, 8, 10),
            (34, 56, 78, 90),
            (),
            (0, 6),
        ]
        dtype_list = [
            torch.bool,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float,
            torch.half,
            torch.double,
        ]
        for list_ in shape_list:
            for dtype in dtype_list:
                x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                x_mlu = x_cpu.to("mlu")
                out_cpu = x_cpu.all()
                out_mlu = x_mlu.all()
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                self.assertTrue(
                    out_mlu.dtype == out_cpu.dtype, "all out dtype is not right"
                )
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

        # not contiguous
        shape_list_nc = [(10, 11, 20), (2, 3, 4, 8, 10), (34, 56, 78, 90), (0, 6)]
        dtype_list_nc = [
            torch.bool,
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float,
            torch.half,
            torch.double,
        ]
        for list_ in shape_list_nc:
            for dtype in dtype_list_nc:
                x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                if x_cpu.dim() == 4:
                    x_cpu = x_cpu.to(memory_format=torch.channels_last)
                x_mlu = x_cpu.to("mlu")
                out_cpu = x_cpu[:, 1:].all()
                out_mlu = x_mlu[:, 1:].all()
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                self.assertTrue(
                    out_mlu.dtype == out_cpu.dtype, "all out dtype is not right"
                )
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_all_out(self):
        shape_list = [
            (10, 11, 20),
            (1111,),
            (2, 3, 4, 8, 10),
            (34, 56, 78, 90),
            (),
            (0, 6),
        ]
        dim_list = [-2, 0, -1, 3, 0, 1]
        dtype_list = [
            (torch.bool, torch.uint8),
            (torch.int8, torch.bool),
            (torch.uint8, torch.uint8),
            (torch.int32, torch.bool),
            (torch.int64, torch.uint8),
            (torch.float, torch.bool),
            (torch.half, torch.uint8),
            (torch.double, torch.bool),
            (torch.int16, torch.uint8),
        ]
        keep_type = [True, False]
        for dim_, list_ in zip(dim_list, shape_list):
            for dtype in dtype_list:
                for keep in keep_type:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype[0])
                    out_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype[1])
                    x_mlu = x_cpu.to("mlu")
                    out_mlu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype[1])
                    out_mlu = out_mlu.to("mlu")
                    torch.all(x_cpu, dim=dim_, keepdim=keep, out=out_cpu)
                    torch.all(x_mlu, dim=dim_, keepdim=keep, out=out_mlu)
                    self.assertTrue(
                        out_mlu.dtype == out_cpu.dtype, "all out dtype is not right"
                    )
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                    )
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

        # not contiguous
        shape_list_nc = [(10, 11, 20), (2, 3, 4, 8, 10), (34, 56, 78, 90), (0, 6)]
        dim_list_nc = [-2, -1, 3, 1]
        dtype_list_nc = [
            (torch.bool, torch.uint8),
            (torch.int8, torch.bool),
            (torch.uint8, torch.uint8),
            (torch.int32, torch.bool),
            (torch.int64, torch.uint8),
            (torch.float, torch.bool),
            (torch.half, torch.uint8),
            (torch.double, torch.bool),
            (torch.int16, torch.uint8),
        ]
        keep_type_nc = [True, False]
        for dim_, list_ in zip(dim_list_nc, shape_list_nc):
            for dtype in dtype_list_nc:
                for keep in keep_type_nc:
                    x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype[0])
                    out_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype[1])[
                        :, 1:
                    ]
                    x_mlu = x_cpu.to("mlu")
                    out_mlu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype[1])[
                        :, 1:
                    ]
                    out_mlu = out_mlu.to("mlu")
                    torch.all(x_cpu[:, 1:], dim=dim_, keepdim=keep, out=out_cpu)
                    torch.all(x_mlu[:, 1:], dim=dim_, keepdim=keep, out=out_mlu)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                    )
                    self.assertTrue(
                        out_mlu.dtype == out_cpu.dtype, "all out dtype is not right"
                    )
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_all_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 10, 24), dtype=torch.bfloat16, device="cpu"
        )
        left_mlu = left.mlu()
        out_cpu = torch.all(left, 1, keepdim=True)
        out_mlu = torch.all(left_mlu, 1, keepdim=True)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        output_cpu = torch.testing.make_tensor(
            out_cpu.shape, dtype=torch.bool, device="cpu"
        )
        output_mlu = output_cpu.mlu()
        torch.all(left, 1, keepdim=True, out=output_cpu)
        torch.all(left_mlu, 1, keepdim=True, out=output_mlu)
        self.assertTensorsEqual(
            output_mlu.cpu().float(), output_cpu.float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            output_mlu.cpu().float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_all_large(self):
        shape_list = [(48, 4096, 13725), (1, 4096 * 48 * 13725)]
        dtype_list = [torch.bool, torch.uint8]
        for list_ in shape_list:
            for dtype in dtype_list:
                x_cpu = (torch.rand(list_, dtype=torch.float) > 0.5).to(dtype)
                x_mlu = x_cpu.to("mlu")
                out_cpu = x_cpu.all()
                out_mlu = x_mlu.all()
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
                self.assertTrue(
                    out_mlu.dtype == out_cpu.dtype, "all out dtype is not right"
                )
                self.assertTrue(out_cpu.size() == out_mlu.size())
                self.assertTrue(out_cpu.stride() == out_mlu.stride())


if __name__ == "__main__":
    run_tests()
