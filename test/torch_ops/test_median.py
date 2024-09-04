from __future__ import print_function

import sys
import os
import itertools
import unittest
import logging

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestMedianOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_median_dim(self):
        type_list = [True, False]
        dtype_list = [torch.half, torch.float]
        shape_list = [(1, 32, 5, 12, 8), (2, 32, 10, 6), (2, 32, 8), (1, 100), (24,)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        param_list = [type_list, dtype_list, shape_list, func_list]
        for test_type, dtype, shape, func in itertools.product(*param_list):
            dim_len = len(shape)
            for dim in range(0, dim_len):
                x = torch.randn(shape, dtype=dtype)
                out_cpu = func(x).float().median(dim, keepdim=test_type)
                out_mlu = func(self.to_mlu(x)).median(dim, keepdim=test_type)
                self.assertTensorsEqual(
                    out_cpu[0].float(), out_mlu[0].cpu().float(), 0.003, use_MSE=True
                )

        # test sliced out
        shape = (2, 32, 10, 6)
        y = torch.randn(shape, dtype=dtype, device="mlu")
        values = torch.randn(shape, dtype=dtype, device="mlu")
        indices = torch.zeros(shape, device="mlu").long() - 1
        torch.median(y, 1, keepdim=False, out=(values[:, 1], indices[:, 1]))
        values_expected, indices_expected = torch.median(y, 1, keepdim=False)

        self.assertTensorsEqual(
            values[:, 1].float().cpu(),
            values_expected.float().cpu(),
            0.003,
            use_MSE=True,
        )
        self.assertTensorsEqual(
            indices[:, 1].cpu(), indices_expected.cpu(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_median(self):
        shape_list = [(1, 32, 5, 12, 8), (2, 128, 10, 6), (2, 512, 8), (1, 100), (24,)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., ::2]]
        dtype_list = [torch.half, torch.float]
        param_list = [shape_list, func_list, dtype_list]
        for shape, func, dtype in itertools.product(*param_list):
            x = torch.randn(shape, dtype=dtype)
            out_cpu = torch.median(func(x.float()))
            out_mlu = torch.median(func(self.to_mlu(x)))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_median_scalar(self):
        x = torch.tensor(5.2, dtype=torch.float)
        out_cpu = torch.median(x)
        out_mlu = torch.median(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_median_exception(self):
        a = torch.randn(167936, 2).to("mlu")
        b = torch.randn(335872, 2).to(torch.half).to("mlu")
        ref_msg = (
            f"MLU median: elements number of input tensor should be less than 167936"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.median(a)
        ref_msg = (
            f"MLU median: elements number of input tensor should be less than 335872"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.median(b)
        ref_msg = "MLU median: elements number of input tensor in the dimension 0 should be less than 41984"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.median(a, dim=0)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_median_bfloat16(self):
        x = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
        out_cpu = torch.median(x)
        out_mlu = torch.median(self.to_mlu(x))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
