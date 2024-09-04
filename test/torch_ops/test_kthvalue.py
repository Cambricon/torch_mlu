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
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue(self):
        keepdim = [True, False, True, False, False]
        shape_list = [
            (2, 3, 4, 2, 1, 7, 8),
            (2, 3, 4),
            (7, 300),
            (20, 26258),
            (3, 4, 5),
        ]
        k_list = [5, 2, 4, 3, 2]
        dim_list = [6, 1, -1, 1, 0]
        type_list = [torch.float, torch.half]
        for t in type_list:
            for i in range(len(shape_list)):  # pylint: disable=C0200
                x = torch.randn(shape_list[i]).to(t)
                out_cpu = torch.kthvalue(x.float(), k_list[i], dim_list[i], keepdim[i])
                out_mlu = torch.kthvalue(
                    self.to_mlu(x), k_list[i], dim_list[i], keepdim[i]
                )
                # kthvalue sorting algorithm for mlu is different from cpu,
                #  when value is the same the kthvalue index may be different,
                # in this case, index test is not included for kthvalue in unit test.
                self.assertTensorsEqual(
                    out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue_channels_last(self):
        shape = (12, 6, 2, 30)
        x = torch.randn(shape).to(memory_format=torch.channels_last)
        out_cpu = torch.kthvalue(x, 1, 3, False)
        out_mlu = torch.kthvalue(self.to_mlu(x), 1, 3, False)
        self.assertTensorsEqual(
            out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue_no_dense(self):
        shape = (12, 6, 2, 30)
        x = torch.randn(shape)
        x_mlu = self.to_mlu(x)[..., ::2]
        x = x[..., ::2]
        out_cpu = torch.kthvalue(x, 1, 3, False)
        out_mlu = torch.kthvalue(x_mlu, 1, 3, False)
        self.assertTensorsEqual(
            out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue_scalar(self):
        input = torch.tensor(3)
        input_mlu = input.to("mlu")
        out_cpu = torch.kthvalue(input, 1)
        out_mlu = torch.kthvalue(input_mlu, 1)
        self.assertTensorsEqual(
            out_cpu[0].float(), out_mlu[0].cpu().float(), 0.0, use_MSE=True
        )
        self.assertTensorsEqual(
            out_cpu[1].float(), out_mlu[1].cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue_empty(self, device="mlu"):
        x = torch.randn((2, 0, 3))
        out_cpu, idx_cpu = torch.kthvalue(x, 1)
        out_mlu, idx_mlu = torch.kthvalue(x.to(device), 1)
        self.assertEqual(out_cpu.shape, out_mlu.shape)
        self.assertEqual(idx_cpu.shape, idx_mlu.shape)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue_out(self):
        shape_list = [(2, 3, 4, 2, 1, 7, 8), (2, 3, 4), (7, 300), (20, 26258), (5, 3)]
        k_list = [5, 2, 4, 3, 1]
        dim_list = [6, 1, -1, 1, 1]
        keepdim = [True, False, True, False, False]
        for i in range(len(shape_list)):  # pylint: disable=C0200
            x = torch.randn(shape_list[i], dtype=torch.float)
            y = torch.randn(shape_list[i], dtype=torch.float)
            index = torch.randn(shape_list[i]).to(torch.long)
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            index_mlu = index.to("mlu")
            if i != len(shape_list) - 1:
                torch.kthvalue(x, k_list[i], dim_list[i], keepdim[i], out=(y, index))
                torch.kthvalue(
                    x_mlu, k_list[i], dim_list[i], keepdim[i], out=(y_mlu, index_mlu)
                )
            else:
                torch.kthvalue(
                    x, k_list[i], dim_list[i], keepdim[i], out=(y[:, 1], index[:, 1])
                )
                torch.kthvalue(
                    x_mlu,
                    k_list[i],
                    dim_list[i],
                    keepdim[i],
                    out=(y_mlu[:, 1], index_mlu[:, 1]),
                )
            self.assertTensorsEqual(y, y_mlu.cpu(), 0.0, use_MSE=True)
            self.assertTensorsEqual(
                index.float(), index_mlu.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue_scalar_out(self):
        input = torch.tensor(3)
        y = torch.tensor(-1)
        index = torch.tensor(-1)
        y_mlu = y.to("mlu")
        index_mlu = index.to("mlu")
        input_mlu = input.to("mlu")
        out_cpu = torch.kthvalue(input, 1, out=(y, index))
        out_mlu = torch.kthvalue(input_mlu, 1, out=(y_mlu, index_mlu))
        self.assertEqual(y.item(), y_mlu.item())
        self.assertEqual(index.item(), index_mlu.item())

    # @unittest.skip("not test")
    @testinfo()
    def test_kthvalue_type(self):
        keepdim = [False]
        shape_list = [(5, 3)]
        k_list = [1]
        dim_list = [1]
        type_list = [
            torch.double,
            torch.float,
            torch.half,
            torch.long,
            torch.int,
            torch.short,
            torch.bool,
        ]
        supported_type_list = [torch.double, torch.float, torch.half]
        for t in type_list:
            i = 0
            x = torch.randn(shape_list[i]).to(t)
            x_mlu = self.to_mlu(x)
            if t not in supported_type_list:
                ref_msg = "not implemented for "
                with self.assertRaisesRegex(RuntimeError, ref_msg):
                    out_mlu = torch.kthvalue(x_mlu, k_list[i], dim_list[i], keepdim[i])
            else:
                out_mlu = torch.kthvalue(x_mlu, k_list[i], dim_list[i], keepdim[i])

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_kthvalue_bfloat16(self):
        shape = [5, 3]
        x = torch.randn(shape).to(torch.bfloat16)
        x_mlu = self.to_mlu(x)
        out_mlu, index_mlu = torch.kthvalue(x_mlu, 1, 1, False)
        out_cpu, index_cpu = torch.kthvalue(x, 1, 1, False)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        self.assertTensorsEqual(index_cpu, index_mlu.cpu(), 0.0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
