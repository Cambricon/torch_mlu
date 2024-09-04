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
    run_tests,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
    TEST_LARGETENSOR,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestTopKOps(TestCase):
    # topk sorting algorithm for mlu is different from cpu,
    # when value is the same the topk index may be different,
    # in this case, index test is not included for topk in unit test.
    # @unittest.skip("not test")
    @testinfo()
    def test_topk(self):
        shape_list = [
            (2, 3, 4, 2, 1, 7, 8),
            (20, 30, 40, 50),
            (20, 1, 3, 5),
            (20, 30, 40, 50, 10),
            (2, 1, 3, 4, 5),
            (2, 3, 4),
            (7, 300),
            (20, 26258),
            (6,),
            (),
            (6, 3, 0, 1),
        ]
        k_list = [5, 2, 1, 20, 1, 2, 4, 3, 1, 1, 3]
        dim_list = [6, 1, 1, 1, 1, 1, -1, 1, 0, 0, 0]
        channel_first = [True, False]
        data_types = [
            torch.float32,
            torch.float16,
            torch.int64,
            torch.int32,
            torch.int8,
            torch.uint8,
        ]
        for i in range(len(shape_list)):
            for channel in channel_first:
                for data_type in data_types:
                    x = torch.randn(shape_list[i], dtype=torch.float).to(data_type)
                    out_cpu = torch.topk(
                        x.float() if data_type == torch.half else x,
                        k_list[i],
                        dim_list[i],
                    )
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_mlu = torch.topk(self.to_mlu(x), k_list[i], dim_list[i])
                    self.assertTensorsEqual(
                        out_cpu[0].float(),
                        out_mlu[0].cpu().float().contiguous(),
                        0.0,
                        use_MSE=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_topk_out(self):
        shape_list = [
            (2, 3, 4, 2, 1, 7, 8),
            (2, 3, 4),
            (7, 300),
            (20, 26258),
            (1, 3, 10),
            (),
            (0, 6),
        ]
        k_list = [5, 2, 4, 3, 4, 1, 3]
        dim_list = [6, 1, -1, -2, -1, -1, -1]
        data_types = [
            torch.float32,
            torch.float16,
            torch.int64,
            torch.int32,
            torch.int8,
            torch.uint8,
        ]
        largest_list = [True, False]
        for i, shape in enumerate(shape_list):
            for data_type in data_types:
                x = torch.randn(shape, dtype=torch.float).to(data_type)
                for l in largest_list:
                    if data_type == torch.half:
                        values_cpu = torch.randn((1)).to(data_type).float()
                    else:
                        values_cpu = torch.randn((1)).to(data_type)

                    values_mlu = torch.randn((1)).to(data_type).to("mlu")
                    indices_cpu = torch.randn((1)).long()
                    indices_mlu = torch.randn((1)).long().to("mlu")
                    torch.topk(
                        x.float() if data_type == torch.half else x,
                        k_list[i],
                        dim_list[i],
                        largest=l,
                        out=(values_cpu, indices_cpu),
                    )
                    torch.topk(
                        self.to_mlu(x),
                        k_list[i],
                        dim_list[i],
                        largest=l,
                        out=(values_mlu, indices_mlu),
                    )
                    self.assertTensorsEqual(
                        values_cpu.float(), values_mlu.cpu().float(), 0.0, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_topk_not_dense(self):
        shape_list = [
            (2, 3, 4, 2, 1, 7, 8 * 2),
            (20, 30, 40, 50 * 2),
            (20, 30, 40, 50, 10 * 2),
            (2, 3, 4 * 2),
            (7, 300 * 2),
            (20, 26258 * 2),
        ]
        k_list = [5, 2, 20, 2, 4, 3]
        dim_list = [6, 1, 1, 1, -1, 1]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            if len(shape_list[i]) == 2:
                x = torch.randn(shape_list[i], dtype=torch.float)[
                    :, : int(shape_list[i][-1] / 2)
                ]
            elif len(shape_list[i]) == 3:
                x = torch.randn(shape_list[i], dtype=torch.float)[
                    :, :, : int(shape_list[i][-1] / 2)
                ]
            elif len(shape_list[i]) == 4:
                x = torch.randn(shape_list[i], dtype=torch.float)[
                    :, :, :, : int(shape_list[i][-1] / 2)
                ]
            elif len(shape_list[i]) == 5:
                x = torch.randn(shape_list[i], dtype=torch.float)[
                    :, :, :, :, : int(shape_list[i][-1] / 2)
                ]
            elif len(shape_list[i]) == 6:
                x = torch.randn(shape_list[i], dtype=torch.float)[
                    :, :, :, :, :, : int(shape_list[i][-1] / 2)
                ]
            elif len(shape_list[i]) == 7:
                x = torch.randn(shape_list[i], dtype=torch.float)[
                    :, :, :, :, :, :, : int(shape_list[i][-1] / 2)
                ]
            out_cpu = torch.topk(x, k_list[i], dim_list[i])
            out_mlu = torch.topk(self.to_mlu(x), k_list[i], dim_list[i])
            self.assertTensorsEqual(
                out_cpu[0].float(),
                out_mlu[0].cpu().float().contiguous(),
                0.0,
                use_MSE=True,
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_topk_bfloat16(self):
        shape_list = [
            (2, 3, 4, 2, 1, 7, 8),
            (20, 30, 40, 50),
            (20, 1, 3, 5),
            (20, 30, 40, 50, 10),
            (2, 1, 3, 4, 5),
            (2, 3, 4),
            (7, 300),
            (20, 26258),
            (6,),
            (),
            (6, 3, 0, 1),
        ]
        k_list = [5, 2, 1, 20, 1, 2, 4, 3, 1, 1, 3]
        dim_list = [6, 1, 1, 1, 1, 1, -1, 1, 0, 0, 0]
        channel_first = [True, False]
        data_types = [torch.bfloat16]
        for i in range(len(shape_list)):
            for channel in channel_first:
                for data_type in data_types:
                    x = torch.randn(shape_list[i], dtype=torch.float).to(data_type)
                    out_cpu = torch.topk(
                        x.float() if data_type == torch.bfloat16 else x,
                        k_list[i],
                        dim_list[i],
                    )
                    if channel is False:
                        x = self.convert_to_channel_last(x)
                    out_mlu = torch.topk(self.to_mlu(x), k_list[i], dim_list[i])
                    self.assertTensorsEqual(
                        out_cpu[0].float(),
                        out_mlu[0].cpu().float().contiguous(),
                        0.0,
                        use_MSE=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_topk_exception(self):
        a = torch.randn((2, 3, 4), dtype=torch.float).bool().to("mlu")
        ref_msg = f"not implemented for 'Bool'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.topk(a, k=2, dim=1)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("21GB")
    def test_topk_large(self):
        shape = (4, 1025, 1024 * 1024)
        dtype = torch.float
        k, dim = 5, -1
        x = torch.randn(shape, dtype=dtype)
        out_cpu = torch.topk(x, k, dim=dim)
        out_mlu = torch.topk(self.to_mlu(x), k, dim=dim)
        self.assertTensorsEqual(
            out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
