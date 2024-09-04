from __future__ import print_function

import unittest
import logging

import sys
import os
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
)  # pylint: disable=C0413, C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestFloorDivideOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_tensor_tensor(self):
        dtype_list = [torch.float, torch.half]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        for dtype in dtype_list:
            for shape1, shape2 in [
                ((), ()),
                ((0, 6), (0, 6)),
                ((12, 0, 12), (1)),
                ((2, 2, 4, 2), (2)),
                ((1, 2), (2, 2, 4, 2)),
                ((2, 1, 2, 4), (1, 2, 4)),
            ]:
                x = torch.randn(shape1, dtype=dtype)
                y = torch.randn(shape2, dtype=dtype)
                out_cpu = torch.floor_divide(x, y)
                out_mlu = torch.floor_divide(x.to("mlu"), y.to("mlu"))
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_tensor_tensor_PYTORCH_11152(self):
        x = torch.randn((1, 4, 1, 64, 64), dtype=torch.float)
        y = torch.randn((1, 4, 1, 64, 64), dtype=torch.float)
        x.as_strided_(x.size(), stride=(4, 1, 4, 256, 4))
        y.as_strided_(y.size(), stride=(16384, 1, 16384, 256, 4))
        out_cpu = torch.floor_divide(x, y)
        out_mlu = torch.floor_divide(x.to("mlu"), y.to("mlu"))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_tensor_scalar(self):
        dtype_list = [torch.double, torch.float, torch.int, torch.long]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        shape_list = [(), (0, 6), (12, 0, 12), (2, 12, 64), (2, 1, 2, 4)]
        scale_list = [1e5, 10, True]
        for t in dtype_list:
            for shape in shape_list:
                for scale in scale_list:
                    x = (torch.randn(shape) * 2).to(t)
                    out_cpu = torch.div(x, scale, rounding_mode="floor")
                    out_mlu = torch.div(x.to("mlu"), scale, rounding_mode="floor")
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_inplace_tensor_tensor(self):
        for shape1, shape2 in [
            ((), ()),
            ((0, 6), (0, 6)),
            ((12, 0, 12), (1)),
            ((2, 2, 4, 2), (2)),
            ((2, 1, 2, 4), (1, 2, 4)),
        ]:
            x = torch.randn(shape1, dtype=torch.float)
            x_mlu = self.to_mlu(x)
            y = torch.randn(shape2, dtype=torch.float)
            y_mlu = self.to_mlu(y)
            ptr1 = x_mlu.data_ptr()
            x.floor_divide_(y)
            x_mlu.floor_divide_(y_mlu)
            ptr2 = x_mlu.data_ptr()
            self.assertEqual(ptr1, ptr2)
            self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_inplace_tensor_scalar(self):
        shape_list = [(), (0, 6), (12, 0, 12), (2, 12, 64), (2, 1, 2, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = self.to_mlu(x)
            ptr1 = x_mlu.data_ptr()
            x.floor_divide_(8.0)
            x_mlu.floor_divide_(8.0)
            ptr2 = x_mlu.data_ptr()
            self.assertEqual(ptr1, ptr2)
            self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_out_tensor_tensor(self):
        for shape1, shape2 in [
            ((), ()),
            ((0, 6), (0, 6)),
            ((12, 0, 12), (1)),
            ((2, 2, 4, 2), (2)),
            ((1, 2), (2, 2, 4, 2)),
            ((2, 1, 2, 4), (1, 2, 4)),
        ]:
            x = torch.randn(shape1, dtype=torch.float)
            x_mlu = self.to_mlu(x)
            y = torch.randn(shape2, dtype=torch.float)
            y_mlu = self.to_mlu(y)
            out_cpu = torch.randn((1))
            out_mlu = torch.randn((1)).to("mlu")
            torch.floor_divide(x, y, out=out_cpu)
            torch.floor_divide(x_mlu, y_mlu, out=out_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            broadcast_shape = torch._C._infer_size(x_mlu.shape, y_mlu.shape)
            out_cpu = torch.randn(broadcast_shape)
            out_mlu = torch.randn(broadcast_shape).to("mlu")
            torch.floor_divide(x, y, out=out_cpu)
            torch.floor_divide(x_mlu, y_mlu, out=out_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_channels_last(self):
        shape = (12, 2, 4, 2)
        x = torch.randn(shape, dtype=torch.float).to(memory_format=torch.channels_last)
        x_mlu = self.to_mlu(x)
        out_cpu = torch.floor_divide(x, 8.0)
        out_mlu = torch.floor_divide(x_mlu, 8.0)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_no_dense(self):
        shape = (12, 2, 1, 2, 4)
        x = torch.randn(shape, dtype=torch.float)
        x_mlu = self.to_mlu(x)[..., ::2]
        x = x[..., ::2]
        out_cpu = torch.floor_divide(x, 8.0)
        out_mlu = torch.floor_divide(x_mlu, 8.0)
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_inplace_channels_last(self):
        shape_list = [(12, 2, 4, 2)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float).to(
                memory_format=torch.channels_last
            )
            x_mlu = self.to_mlu(x)
            x.floor_divide_(0.1)
            x_mlu_ptr = x_mlu.data_ptr()
            x_mlu.floor_divide_(0.1)
            self.assertEqual(x_mlu_ptr, x_mlu.data_ptr(), 0)
            self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_inplace_no_dense(self):
        shape_list = [(12, 2, 1, 2, 4)]
        for shape in shape_list:
            x = torch.randn(shape, dtype=torch.float)
            x_mlu = self.to_mlu(x)
            x_mlu_ptr = x_mlu.data_ptr()
            x = x[::2]
            x.floor_divide_(0.1)
            x_mlu = x_mlu[::2]
            x_mlu.floor_divide_(0.1)
            self.assertEqual(x_mlu_ptr, x_mlu.data_ptr(), 0)
            self.assertTensorsEqual(x.float(), x_mlu.cpu().float(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_mixed_memory_format(self):
        x = torch.randn((3, 1, 5, 1), dtype=torch.float).to(
            memory_format=torch.channels_last
        )
        y = torch.randn((4, 5, 1), dtype=torch.float)
        out_cpu = torch.floor_divide(x, y)
        out_mlu = torch.floor_divide(self.to_mlu(x), self.to_mlu(y))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )
        out_cpu = torch.floor_divide(y, x)
        out_mlu = torch.floor_divide(self.to_mlu(y), self.to_mlu(x))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_mixed_dtype(self):
        dtype_list = [
            torch.float,
            torch.half,
            torch.int,
            torch.short,
            torch.long,
            torch.int8,
            torch.bool,
            torch.uint8,
        ]
        dtype_list += [torch.bfloat16] if TEST_BFLOAT16 else []
        shape = (12, 2, 1, 2, 4)
        for data_type in dtype_list:
            x = torch.randn(shape, dtype=torch.float).to(data_type)
            y = torch.randn(shape, dtype=torch.float)
            out_cpu = torch.floor_divide(x, y)
            out_mlu = torch.floor_divide(self.to_mlu(x), self.to_mlu(y))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_floor_divide_exception(self):
        x_mlu = (torch.rand(1, 16) * 10).to("mlu")
        y_mlu = (torch.rand(16, 1) * 10).to("mlu")
        ref_msg = (
            r"output with shape \[1, 16\] doesn't match the broadcast shape \[16, 16\]"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.floor_divide_(y_mlu)

    @testinfo()
    def test_floor_divide_infnan(self):
        x = torch.tensor([float("inf"), float("nan")]).to("mlu")
        y = torch.randn([1, 2]).to("mlu")
        out_mlu = torch.floor_divide(x, y)
        self.assertTrue(torch.isnan(out_mlu).all())

        x2 = torch.tensor([0.0703], dtype=torch.half)
        y2 = torch.tensor([0.0], dtype=torch.half)
        out_cpu = torch.floor_divide(x2, y2)
        out_mlu_2 = torch.floor_divide(x2.to("mlu"), y2.to("mlu"))
        torch.testing.assert_close(out_cpu, out_mlu_2.cpu(), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
