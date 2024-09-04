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
    TestCase,
    read_card_info,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0411, C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestLogicalNotOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_logical_not(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape in [
                (),
                (1),
                (256, 144, 7, 15, 2, 1),
                (256, 7),
                (5),
                (2, 3, 4),
                (1, 117, 1, 4, 1, 5, 1, 2),
                (256, 144, 7, 15, 2, 1, 1, 1),
            ]:
                x = torch.randn(shape).to(t)
                out_cpu = torch.logical_not(x)
                out_mlu = torch.logical_not(self.to_mlu(x))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_not_channels_last(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape in [(12, 15, 18, 26), (20, 144, 8, 30)]:
                x = torch.randn(shape).to(t)
                x_cl = self.convert_to_channel_last(x)
                out_cpu = torch.logical_not(x_cl)
                x_mlu_cl = self.convert_to_channel_last(self.to_mlu(x))
                out_mlu = torch.logical_not(x_mlu_cl)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_not_inplace(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape in [
                (),
                (1),
                (256, 144, 7, 15, 2, 1),
                (256, 7),
                (5),
                (2, 3, 4),
                (1, 117, 1, 4, 1, 5, 1, 2),
                (256, 144, 7, 15, 2, 1, 1, 1),
            ]:
                x = torch.randn(shape).to(t)
                x_mlu = x.to("mlu")
                x_mlu_data = x_mlu.data_ptr()
                x.logical_not_()
                x_mlu.logical_not_()
                self.assertEqual(x_mlu_data, x_mlu.data_ptr())
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logical_not_out(self):
        type_list = [
            torch.bool,
            torch.float,
            torch.int,
            torch.short,
            torch.int8,
            torch.uint8,
            torch.long,
        ]
        for t in type_list:
            for shape in [
                (),
                (1),
                (256, 144, 7, 15, 2, 1),
                (256, 7),
                (5),
                (2, 3, 4),
                (1, 117, 1, 4, 1, 5, 1, 2),
                (256, 144, 7, 15, 2, 1, 1, 1),
            ]:
                x = torch.randn(shape).to(t)
                broadcast_shape = torch._C._infer_size(x.shape, x.shape)
                out_cpu = torch.zeros(broadcast_shape, dtype=torch.bool)
                out_mlu = torch.zeros(broadcast_shape, dtype=torch.bool).to("mlu")
                torch.logical_not(x, out=out_cpu)
                torch.logical_not(self.to_mlu(x), out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_logical_not_bfloat16(self):
        left = torch.testing.make_tensor(
            (2, 3, 4, 6), dtype=torch.bfloat16, device="cpu"
        )
        out_cpu = torch.logical_not(left)
        out_mlu = torch.logical_not(left.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("48GB")
    def test_logical_not_large(self):
        left = torch.testing.make_tensor(
            (5, 1024, 1024, 1024), dtype=torch.float, device="cpu"
        )
        out_cpu = torch.logical_not(left)
        out_mlu = torch.logical_not(left.mlu())
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
