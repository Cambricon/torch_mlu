from __future__ import print_function

import sys
import os
import unittest
import logging
import copy
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestRoundOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_round(self):
        shape_list = [
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
        ]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.round(x)
            out_mlu = torch.round(self.to_mlu(x))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_channel_last(self):
        shape = (2, 128, 10, 6)
        x = torch.randn(shape, dtype=torch.float).to(memory_format=torch.channels_last)
        out_cpu = torch.round(x)
        out_mlu = torch.round(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_not_dense(self):
        shape_list = [(2, 3, 4), (1, 32, 5, 12, 8), (2, 128, 10, 6)]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out_cpu = torch.round(x[:, ..., :2])
            out_mlu = torch.round(self.to_mlu(x)[:, ..., :2])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_inplace(self):
        shape_list = [
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
        ]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to("mlu")
            out_cpu = torch.round_(x_cpu)
            out_mlu = torch.round_(x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to("mlu")
            x_cpu.round_()
            x_mlu.round_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_inplace_channel_last(self):
        shape_list = [(32, 5, 12, 8), (2, 128, 10, 6)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i]).to(memory_format=torch.channels_last)
            x_mlu = x_cpu.to("mlu")
            out_cpu = torch.round_(x_cpu)
            out_mlu = torch.round_(x_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

            x_cpu = torch.randn(shape_list[i]).to(memory_format=torch.channels_last)
            x_mlu = x_cpu.to("mlu")
            x_cpu.round_()
            x_mlu.round_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_inplace_not_dense(self):
        shape_list = [(2, 3, 4), (1, 32, 5, 12, 8), (2, 128, 10, 6)]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to("mlu")
            out_cpu = torch.round_(x_cpu[:, ..., :2])
            out_mlu = torch.round_(x_mlu[:, ..., :2])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to("mlu")
            x_cpu[:, ..., :2].round_()
            x_mlu[:, ..., :2].round_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_out(self):
        shape_list = [
            (2, 3, 4, 3, 4, 2, 1),
            (2, 3, 4),
            (1, 32, 5, 12, 8),
            (2, 128, 10, 6),
            (2, 512, 8),
            (1, 100),
            (24,),
        ]
        for i, _ in enumerate(shape_list):
            x_cpu = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = x_cpu.to("mlu")
            out_tmpcpu = torch.zeros(shape_list[i])
            out_tmpmlu = torch.zeros(shape_list[i]).to("mlu")
            out_tmpcpu_2 = torch.zeros((1))
            out_tmpmlu_2 = torch.zeros((1)).to("mlu")
            out_cpu = torch.round(x_cpu, out=out_tmpcpu)
            out_mlu = torch.round(x_mlu, out=out_tmpmlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
            self.assertTensorsEqual(out_tmpcpu, out_tmpmlu.cpu(), 0)
            out_cpu_2 = torch.round(x_cpu, out=out_tmpcpu_2)
            out_mlu_2 = torch.round(x_mlu, out=out_tmpmlu_2)
            self.assertTensorsEqual(out_cpu_2, out_mlu_2.cpu(), 0)
            self.assertTensorsEqual(out_tmpcpu_2, out_tmpmlu_2.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.randn(shape_list[i], dtype=torch.float)
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).mlu()
            out_mlu = copy.deepcopy(out).mlu()
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.round(x, out=out)
            torch.round(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_round_dtype(self):
        dtype_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.long,
            torch.int32,
            torch.int16,
            torch.uint8,
            torch.int8,
        ]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half).to(dtype)
            x_mlu = x.mlu()
            x = x.float()
            x.round_()
            x_mlu.round_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.0, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_round_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        for i, _ in enumerate(shape_list):
            x = torch.randn(shape_list[i], dtype=torch.half)
            x_mlu = self.to_mlu(x)
            x_cpu = x.float()
            x_cpu.round_()
            x_mlu.round_()
            self.assertTensorsEqual(x_cpu, x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_round_bfloat16(self):
        x = torch.randn((2, 3, 4, 5, 6), dtype=torch.bfloat16)
        x_mlu = x.mlu()
        x = x.float()
        x.round_()
        x_mlu.round_()
        self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.0, use_MSE=True)


if __name__ == "__main__":
    run_tests()
