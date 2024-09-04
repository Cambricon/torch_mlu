from __future__ import print_function

import sys
import logging
import os
import unittest
from itertools import product

import torch
import torch_mlu

from torch.testing import make_tensor

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0411, C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestConstantPadNdOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_constant_pad_nd(self):
        shape_list = [(2, 3, 4), (12, 32, 64, 64), (12, 3, 4, 5, 2)]
        pad_list = [
            (1, 2),
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (-1, 2, -2, 3),
            (1, 1, 1, 2, 2, 2),
        ]

        # skip torch.long and torch.double, maybe overflow
        dtype_list = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.float,
            torch.half,
            torch.bool,
        ]
        func_list = [self.convert_to_channel_last, self.to_non_dense, lambda x: x]
        loop_list = [shape_list, pad_list, dtype_list, func_list]
        for shape, pad, dtype, func in product(*loop_list):
            x = make_tensor(shape, dtype=dtype, device="cpu")
            out_cpu = torch.constant_pad_nd(func(x), pad, 0)
            out_mlu = torch.constant_pad_nd(func(x).mlu(), pad, 0)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_constant_pad_zero_dim(self):
        m = torch.nn.ConstantPad2d(-1, 3.5)
        a = torch.randn(1, 2, 2)
        out_cpu = m(a)
        out_mlu = m(a.mlu())
        self.assertEqual(out_cpu.float(), out_mlu.cpu().float())

    # @unittest.skip("not test")
    @testinfo()
    def test_constant_pad_nd_limit_test(self):
        # 1729382256910270500 will cast float32 172938225691027046400
        dtype_value_list = [
            (torch.float32, 1729382256910270500),
            # (torch.float64, 1729382256910270500),
            (torch.int32, 2**30)
            # mlu memory only support int32, so skip
            # (torch.int64, 1729382256910270500)
        ]
        for dtype_value in dtype_value_list:
            x = torch.rand((2, 3, 4)).to(dtype=dtype_value[0])
            out_cpu = torch.constant_pad_nd(x, (1, 2), dtype_value[1])
            out_mlu = torch.constant_pad_nd(x.mlu(), (1, 2), dtype_value[1])
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.float().cpu(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_constant_pad_nd_issue(self):
        # PYTORCH-10369
        pad = (0, 0, 1, 0)
        input = torch.randn(2, 1, 6, 3)
        value = 0

        def to_non_dense(data, dim=None, distance=2):
            if not type(data) == torch.Tensor:
                print(
                    "[Warning]: It's not available to convert an unknown object to non-dense type"
                )
                return data
            # convert the last channel as default.
            convert_dim = data.dim()
            if dim is not None:
                convert_dim = dim
            if convert_dim > data.dim():
                print(
                    f"[Warning]: The max available expand dim for a {data.dim()} Tensor"
                    f" is {data.dim()}, but got specified dim as {dim}."
                )
                convert_dim = data.dim()
            a = data.unsqueeze(convert_dim)
            b = torch.cat([a for _ in range(distance)], convert_dim)
            return b.select(dim=convert_dim, index=0)

        cpu_out = torch.ops.aten.constant_pad_nd(input, pad, value)
        device_out = torch.ops.aten.constant_pad_nd(
            to_non_dense(input.mlu().to(memory_format=torch.channels_last), 2),
            pad,
            value,
        )
        torch.testing.assert_close(cpu_out, device_out.cpu())

    # @unittest.skip("not test")
    @testinfo()
    def test_constant_pad_nd_exception(self):
        with self.assertRaisesRegex(
            RuntimeError, "Length of pad must be even but instead it equals"
        ):
            x = torch.randn((1, 2, 4), dtype=torch.float32).mlu()
            out_mlu = torch.constant_pad_nd(x, (1,), 0.1)

        with self.assertRaisesRegex(
            RuntimeError, "Length of pad should be no more than twice the number of"
        ):
            x = torch.randn((2, 4), dtype=torch.float32).mlu()
            out_mlu = torch.constant_pad_nd(x, (1, 2, 1, 1, 1, 1), 0.1)

        with self.assertRaisesRegex(RuntimeError, "resulted in a negative output size"):
            x = torch.randn((1, 1), dtype=torch.float32).mlu()
            out_mlu = torch.constant_pad_nd(x, (-1, -1), 0.1)

        with self.assertRaisesRegex(RuntimeError, "MLU constant_pad_nd"):
            x = torch.randn((1, 2, 4), dtype=torch.complex64).mlu()
            out_mlu = torch.constant_pad_nd(x, (1, 2), 0.1)

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_constant_pad_nd_large_exceptions(self):
        shape = [48, 4096, 13725]
        pad = [1, 2]
        x = make_tensor(shape, dtype=torch.float, device="cpu")
        out_cpu = torch.constant_pad_nd(x, pad, 0)
        out_mlu = torch.constant_pad_nd(x.mlu(), pad, 0)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @skipBFloat16IfNotSupport()
    @testinfo()
    def test_constant_pad_nd_bfloat16(self):
        shape = [2, 3, 4]
        pad = [1, 2]
        x = torch.randn(shape, dtype=torch.bfloat16, device="cpu")
        out_cpu = torch.constant_pad_nd(x, pad, 0)
        out_mlu = torch.constant_pad_nd(x.mlu(), pad, 0)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
