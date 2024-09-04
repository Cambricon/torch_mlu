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
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestEyeOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_tensor_empty(self, device="mlu"):
        self.assertEqual((0, 0), torch.eye(0, device=device).shape)
        self.assertEqual((0, 0), torch.eye(0, 0, device=device).shape)
        self.assertEqual((5, 0), torch.eye(5, 0, device=device).shape)
        self.assertEqual((0, 5), torch.eye(0, 5, device=device).shape)

    # @unittest.skip("not test")
    @testinfo()
    def test_eye(self):
        for dtype in [
            torch.uint8,
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float64,
            torch.float32,
            torch.float16,
        ]:
            for n, m in product([3, 5, 7], repeat=2):
                # Construct identity using diagonal and fill
                res1 = torch.eye(n, m, device="mlu", dtype=dtype)
                naive_eye = torch.zeros(n, m, dtype=dtype)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertTensorsEqual(naive_eye.float(), res1.cpu().float(), 0)

                # Check eye_m_out outputs
                res2 = torch.empty(0, device="mlu", dtype=dtype)
                torch.eye(n, m, out=res2)
                self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), 0)

                # Check eye_out outputs
                res_cpu = torch.eye(n, device="cpu", dtype=dtype)
                res_mlu = torch.empty(0, device="mlu", dtype=dtype)
                torch.eye(n, out=res_mlu)
                self.assertTensorsEqual(res_cpu.float(), res_mlu.cpu().float(), 0)

                # out with slice.
                out_cpu = torch.zeros((10, 10), dtype=dtype)
                out_mlu = copy.deepcopy(out_cpu).to("mlu")
                for outsize in ((1, 1), (n, m), (10, 10)):
                    torch.eye(n, m, out=out_cpu[0 : outsize[0], 0 : outsize[1]])
                    torch.eye(n, m, out=out_mlu[0 : outsize[0], 0 : outsize[1]])
                    self.assertTensorsEqual(out_cpu.float(), out_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_eye_exception(self):
        a = torch.randn(9).reshape(3, 3).to("mlu")
        ref_msg = "n must be greater or equal to 0, got -1"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.eye(-1, out=a)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_eye_bfloat16(self):
        dtype = torch.bfloat16
        for n, m in product([3, 5, 7], repeat=2):
            # Construct identity using diagonal and fill
            res1 = torch.eye(n, m, device="mlu", dtype=dtype)
            naive_eye = torch.zeros(n, m, dtype=dtype)
            naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
            self.assertTensorsEqual(naive_eye.float(), res1.cpu().float(), 0)

            # Check eye_m_out outputs
            res2 = torch.empty(0, device="mlu", dtype=dtype)
            torch.eye(n, m, out=res2)
            self.assertTensorsEqual(res1.cpu().float(), res2.cpu().float(), 0)

            # Check eye_out outputs
            res_cpu = torch.eye(
                n, device="cpu", dtype=torch.float
            )  # eye on cpu not support bf16
            res_mlu = torch.empty(0, device="mlu", dtype=dtype)
            torch.eye(n, out=res_mlu)
            self.assertTensorsEqual(res_cpu.float(), res_mlu.cpu().float(), 0)

            # out with slice.
            out_cpu = torch.zeros((10, 10), dtype=torch.float)
            out_mlu = copy.deepcopy(out_cpu).to("mlu").to(dtype)
            for outsize in ((1, 1), (n, m), (10, 10)):
                torch.eye(n, m, out=out_cpu[0 : outsize[0], 0 : outsize[1]])
                torch.eye(n, m, out=out_mlu[0 : outsize[0], 0 : outsize[1]])


if __name__ == "__main__":
    unittest.main()
