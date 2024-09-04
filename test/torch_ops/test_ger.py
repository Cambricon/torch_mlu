import sys
import os
import unittest
import logging
from itertools import product

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestGerOp(TestCase):
    def run_ger_test(self, v0, v1):
        for dtype in [torch.float32, torch.float16]:
            out_mlu = torch.ger(
                v0.to(dtype=dtype).to("mlu"), v1.to(dtype=dtype).to("mlu")
            )
            out_cpu = torch.ger(v0.float(), v1.float())
            self.assertTensorsEqual(out_mlu.cpu().float(), out_cpu, 0.003, use_MSE=True)

    def run_ger_out_test(self, v0, v1):
        for dtype in [torch.float32, torch.float16]:
            zero_mlu = torch.zeros(v0.size(0), v1.size(0), dtype=dtype, device="mlu")
            zero_cpu = torch.zeros(
                v0.size(0), v1.size(0), dtype=torch.float, device="cpu"
            )
            out_mlu = torch.ger(
                v0.to(dtype=dtype).to("mlu"), v1.to(dtype=dtype).to("mlu"), out=zero_mlu
            )
            out_cpu = torch.ger(v0.float(), v1.float(), out=zero_cpu)
            self.assertTensorsEqual(out_mlu.cpu().float(), out_cpu, 0.003, use_MSE=True)
            self.assertTensorsEqual(
                zero_mlu.cpu().float(), zero_cpu, 0.003, use_MSE=True
            )
            self.assertTensorsEqual(out_cpu, zero_cpu, 0.003, use_MSE=True)

    def run_ger_out_not_contiguous_test(self, v0, v1):
        func_list = [lambda x: x, lambda x: x[..., ::2]]
        list_list = [func_list, func_list, func_list]
        for v0_func, v1_func, out_func in product(*list_list):
            zero_mlu = out_func(torch.zeros(v0.size(0), v1.size(0), device="mlu"))
            zero_cpu = out_func(torch.zeros(v0.size(0), v1.size(0), device="cpu"))
            out_mlu = torch.ger(
                v0_func(v0.to("mlu")), v1_func(v1.to("mlu")), out=zero_mlu
            )
            out_cpu = torch.ger(v0_func(v0), v1_func(v1), out=zero_cpu)
            self.assertTensorsEqual(out_mlu.cpu().float(), out_cpu, 0.003, use_MSE=True)
            self.assertTensorsEqual(
                zero_mlu.cpu().float(), zero_cpu, 0.003, use_MSE=True
            )
            self.assertTensorsEqual(out_cpu, zero_cpu, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_ger_order(self):
        v0 = torch.arange(10)
        v1 = torch.arange(1, 11)
        self.run_ger_test(v0, v1)

    # @unittest.skip("not test")
    @testinfo()
    def test_ger(self):
        shape_list = [((0,), (0)), ((5,), (0,)), ((0,), (5,)), ((10,), (10,))]
        for shape_vec in shape_list:
            v0 = torch.randn(shape_vec[0], dtype=torch.float)
            v1 = torch.randn(shape_vec[1], dtype=torch.float)
            v0_shape = v0.shape
            v1_shape = v1.shape
            self.run_ger_test(v0, v1)
            self.assertEqual(v0.shape, v0_shape)
            self.assertEqual(v1.shape, v1_shape)

        # Tests 0-strided
        v0 = torch.randn(1, dtype=torch.float).expand(10)
        v1 = torch.randn(10, dtype=torch.float)
        self.run_ger_test(v0, v1)

    # @unittest.skip("not test")
    @testinfo()
    def test_ger_out(self):
        v0 = torch.randn(10, dtype=torch.float)
        v1 = torch.randn(10, dtype=torch.float)
        self.run_ger_out_test(v0, v1)

        # Tests 0-strided
        v0 = torch.randn(1, dtype=torch.float).expand(10)
        v1 = torch.randn(10, dtype=torch.float)
        self.run_ger_out_test(v0, v1)

    # @unittest.skip("not test")
    @testinfo()
    def test_ger_out_not_contiguous(self):
        v0 = torch.randn(10, dtype=torch.float)
        v1 = torch.randn(10, dtype=torch.float)
        self.run_ger_out_not_contiguous_test(v0, v1)

        # Tests 0-strided
        v0 = torch.randn(1, dtype=torch.float).expand(10)
        v1 = torch.randn(10, dtype=torch.float)
        self.run_ger_out_not_contiguous_test(v0, v1)

    # @unittest.skip("not test")
    @testinfo()
    def test_ger_invalid_shape(self):
        a = torch.randn(2, 3).to("mlu")
        b = torch.randn(2, 3).to("mlu")
        ref_msg = "outer: Expected 1-D argument self, but got 2-D"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.ger(a, b)


if __name__ == "__main__":
    unittest.main()
