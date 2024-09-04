from __future__ import print_function

import sys
import os
import unittest
import logging

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)

logging.basicConfig(level=logging.DEBUG)
torch.manual_seed(1234)


class TestAnyOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_any_dim(self):
        shape_list = [
            (10,),
            (3, 5),
            (4, 5, 8),
            (8, 10, 12, 14),
            (8, 0, 12, 14),
            (0,),
            (0, 5),
            (4, 5, 0),
            (8, 0, 12, 14),
        ]
        dim_list = [0, 1, -1, 3, 3, 0, 1, -1, 3]
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
        for shape, dim, dtype in zip(shape_list, dim_list, dtype_list):
            x = torch.rand(shape, dtype=torch.float).to(dtype)
            if x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)
            x_1 = x < 0.05
            out_cpu_1 = x_1.any()
            out_cpu_2 = x_1.any(dim)
            out_mlu_1 = x_1.to("mlu").any()
            out_mlu_2 = x_1.to("mlu").any(dim)
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
            self.assertTrue(
                out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right"
            )
            # Bool Result diff: 0.0
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(), 0.0, use_MSE=True)
            self.assertTrue(
                out_mlu_2.dtype == out_cpu_2.dtype, "any out dtype is not right"
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_any(self):
        shape_list = [
            (10,),
            (3, 5),
            (4, 5, 8),
            (8, 10, 12, 14),
            (0,),
            (0, 5),
            (4, 5, 0),
            (8, 0, 12, 14),
        ]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float)
            if x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)
            x_1 = x < 0.05
            out_cpu_1 = x_1.any()
            out_mlu_1 = x_1.to("mlu").any()
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
            self.assertTrue(
                out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right"
            )
            # Bool Result diff: 0.0

            self.assertTrue(out_cpu_1.size() == out_mlu_1.size())
            self.assertTrue(out_cpu_1.stride() == out_mlu_1.stride())
        x_1 = torch.tensor(True)
        out_cpu_1 = x_1.any()
        out_mlu_1 = x_1.to("mlu").any()
        self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
        self.assertTrue(
            out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right"
        )
        x_1 = torch.tensor(False)
        out_cpu_1 = x_1.any()
        out_mlu_1 = x_1.to("mlu").any()
        self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
        self.assertTrue(
            out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right"
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_any_not_contiguous(self):
        shape_list = [(100, 200), (99, 30, 40), (34, 56, 78, 90)]
        dim_list = [-2, 1, 2]
        for i, list_ in enumerate(shape_list):
            x = torch.rand(list_, dtype=torch.float)
            if x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)
            x_1 = x.round().bool()
            out_cpu_1 = x_1[:, 1:].any()
            out_cpu_2 = x_1[:, 1:].any(dim_list[i])
            out_mlu_1 = x_1.to("mlu")[:, 1:].any()
            out_mlu_2 = x_1.to("mlu")[:, 1:].any(dim_list[i])
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(), 0.003, use_MSE=True)
            self.assertTrue(out_cpu_1.size() == out_mlu_1.size())
            self.assertTrue(out_cpu_2.size() == out_mlu_2.size())
            self.assertTrue(out_cpu_1.stride() == out_mlu_1.stride())
            self.assertTrue(out_cpu_2.stride() == out_mlu_2.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_any_out(self):
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
                    torch.any(x_cpu, dim=dim_, keepdim=keep, out=out_cpu)
                    torch.any(x_mlu, dim=dim_, keepdim=keep, out=out_mlu)
                    self.assertTrue(
                        out_mlu.dtype == out_cpu.dtype, "any out dtype is not right"
                    )
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                    )
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

    # @unittest.skip("not test")
    @testinfo()
    def test_any_out_not_contiguous(self):
        shape_list_nc = [(10, 11, 20), (2, 3, 4, 8, 10), (34, 56, 78, 90), (0, 6)]
        dim_list_nc = [-2, -1, 3, 1]
        dtype_list_nc = [
            (torch.bool, torch.bool),
            (torch.int8, torch.uint8),
            (torch.uint8, torch.bool),
            (torch.int32, torch.uint8),
            (torch.int64, torch.bool),
            (torch.float, torch.uint8),
            (torch.half, torch.bool),
            (torch.double, torch.uint8),
            (torch.int16, torch.bool),
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
                    out_mlu = (
                        (torch.rand(list_, dtype=torch.float) > 0.5)
                        .mlu()
                        .to(dtype[1])[:, 1:]
                    )
                    torch.any(x_cpu[:, 1:], dim=dim_, keepdim=keep, out=out_cpu)
                    torch.any(x_mlu[:, 1:], dim=dim_, keepdim=keep, out=out_mlu)
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                    )
                    self.assertTrue(
                        out_mlu.dtype == out_cpu.dtype, "any out dtype is not right"
                    )
                    self.assertTrue(out_cpu.size() == out_mlu.size())
                    self.assertTrue(out_cpu.stride() == out_mlu.stride())

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_all_bfloat16(self):
        left_cpu = torch.testing.make_tensor(
            (2, 10, 24), dtype=torch.bfloat16, device="cpu"
        )
        left_mlu = left_cpu.mlu()
        out_cpu = torch.any(left_cpu, 1, keepdim=True)
        out_mlu = torch.any(left_mlu, 1, keepdim=True)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )
        output_cpu = torch.testing.make_tensor(
            out_cpu.shape, dtype=torch.bool, device="cpu"
        )
        output_mlu = output_cpu.mlu()
        torch.any(left_cpu, 1, keepdim=True, out=output_cpu)
        torch.any(left_mlu, 1, keepdim=True, out=output_mlu)
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
    def test_any_large(self):
        shape_list = [(48, 4096, 13725), (1, 4096 * 48 * 13725)]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float)
            if x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)
            x_1 = x < 0.05
            out_cpu_1 = x_1.any()
            out_mlu_1 = x_1.to("mlu").any()
            self.assertTensorsEqual(out_cpu_1, out_mlu_1.cpu(), 0.0, use_MSE=True)
            self.assertTrue(
                out_mlu_1.dtype == out_cpu_1.dtype, "any out dtype is not right"
            )
            # Bool Result diff: 0.0

            self.assertTrue(out_cpu_1.size() == out_mlu_1.size())
            self.assertTrue(out_cpu_1.stride() == out_mlu_1.stride())


if __name__ == "__main__":
    run_tests()
