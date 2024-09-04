import sys
import logging
import os
import copy
import unittest

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestFracOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_frac(self):
        shape_list = [
            (1, 3, 224, 224),
            (2, 3, 4),
            (2, 2),
            (254, 254, 112, 1, 1, 3),
            (0, 2, 3),
        ]
        dtypes = [torch.float, torch.double, torch.half]
        for shape in shape_list:
            for t in dtypes:
                cpu_type = t
                if t == torch.half:
                    cpu_type = torch.float32
                x = torch.randn(shape, dtype=torch.float).to(t)
                out_cpu = torch.frac(x.to(cpu_type))
                out_mlu = torch.frac(self.to_mlu(x))
                if t == torch.half:
                    out_mlu = out_mlu.to(torch.float32)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_scalar(self):
        x_0 = torch.tensor(-1.57)
        out_cpu = torch.frac(x_0)
        out_mlu = torch.frac(self.to_mlu(x_0))
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_inplace(self):
        shape_list = [(1, 3, 224, 224), (2, 3, 4), (2, 2), (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = self.to_mlu(x)
            y_data = y.data_ptr()
            torch.frac_(x)
            torch.frac_(y)
            self.assertEqual(y_data, y.data_ptr())
            self.assertTensorsEqual(x, y.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_t(self):
        shape_list = [(1, 3, 224, 224), (2, 3, 4), (2, 2), (254, 254, 112, 1, 1, 3)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = x.frac()
            out_mlu = self.to_mlu(x).frac()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_out(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
            (1000),
            (),
        ]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = torch.randn(1, dtype=torch.float)
            y_mlu = copy.deepcopy(y).to(torch.device("mlu"))

            torch.frac(x, out=y)
            torch.frac(self.to_mlu(x), out=y_mlu)
            self.assertTensorsEqual(y, y_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_channelslast(self):
        shape_list = [(64, 3, 6, 6), (2, 25, 64, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)

            # channels_last input
            out_cpu = torch.frac(x.to(memory_format=torch.channels_last))
            out_mlu = torch.frac(x.to("mlu").to(memory_format=torch.channels_last))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_not_dense(self):
        shape_list = [(64, 3, 6, 6), (2, 25, 64, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            y = x.to("mlu")

            # not_dense input
            out_cpu = torch.frac(x[..., :2])
            out_mlu = torch.frac(y[..., :2])
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_permute(self):
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
            x_mlu = copy.deepcopy(x).to("mlu")
            out_mlu = copy.deepcopy(out).to("mlu")
            x, out = x.permute(permute_shape[i]), out.permute(permute_shape[i])
            x_mlu, out_mlu = x_mlu.permute(permute_shape[i]), out_mlu.permute(
                permute_shape[i]
            )
            torch.frac(x, out=out)
            torch.frac(x_mlu, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_frac_dtype(self):
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype in dtype_list:
            x = torch.randn((2, 3, 4, 5, 6), dtype=torch.half)
            x_mlu = self.to_mlu_dtype(x, dtype)
            x = x.float()
            x.frac_()
            x_mlu.frac_()
            self.assertTensorsEqual(x, x_mlu.cpu().float(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_frac_bfloat16(self):
        x = torch.randn((2, 3, 4, 5, 6), dtype=torch.bfloat16)
        x_mlu = self.to_mlu(x)
        out_cpu = x.frac()
        out_mlu = x_mlu.frac()
        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
