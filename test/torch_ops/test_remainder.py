from __future__ import print_function

import unittest
import logging
import copy
import sys
import os
from itertools import product
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)


class TestRemainderOp(TestCase):  # pylint: disable=R0904
    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_tensor_tensor(self):
        dtype_list = [
            (torch.double, 3e-3),
            (torch.float, 3e-3),
            (torch.half, 3e-3),
            (torch.int32, 0),
            (torch.short, 0),
            (torch.long, 0),
        ]
        for data_type, err in dtype_list:
            for shape1, shape2 in [
                ((), ()),
                ((6), (6)),
                ((6, 0, 3), (6, 0, 1)),
                ((1, 3, 224, 224), (1, 3, 224, 1)),
                ((2, 2, 4, 2), (2)),
                ((1, 2), (2, 2, 4, 2)),
                ((2, 1, 2, 4), (1, 2, 4)),
                ((1, 3, 16, 1), (1, 1, 1, 16)),
            ]:
                x = (20 * torch.randn(shape1, dtype=torch.float)).to(data_type)
                y = (20 * torch.rand(shape2, dtype=torch.float)).to(data_type) + 1
                x_mlu = x.mlu()
                y_mlu = y.mlu()
                if data_type == torch.half:
                    out_cpu = x.float() % y.float()
                else:
                    out_cpu = x % y
                out_mlu = x_mlu % y_mlu
                self.assertEqual(
                    out_cpu.dtype if data_type != torch.half else torch.half,
                    out_mlu.dtype,
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_backward(self):
        parameter_list = [
            (torch.double, 3e-3, False),
            (torch.float, 3e-3, True),
            (torch.half, 3e-3, False),
        ]
        shape_list = [
            ((), ()),
            ((6, 0, 3), (6, 0, 1)),
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 2, 4, 2), (2)),
        ]
        for parameter, shape in product(parameter_list, shape_list):
            data_type, err, other_is_scalar = parameter
            shape1, shape2 = shape
            x = (20 * torch.randn(shape1, dtype=torch.float)).to(data_type)
            y = (
                (20 * torch.rand(shape2, dtype=torch.float)).to(data_type) + 1
                if other_is_scalar is False
                else 2.0
            )
            x_cpu = torch.nn.Parameter(x)
            y_cpu = y
            x_mlu = torch.nn.Parameter(x.mlu())
            y_mlu = y.mlu() if other_is_scalar is False else y
            if data_type == torch.half:
                out_cpu = x_cpu.float() % y_cpu.float()
            else:
                out_cpu = x_cpu % y_cpu
            out_mlu = x_mlu % y_mlu
            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.mlu()
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertEqual(
                out_cpu.dtype if data_type != torch.half else torch.half, out_mlu.dtype
            )
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )
            self.assertTensorsEqual(
                x_cpu.grad.float(), x_mlu.grad.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_tensor_tensor_channels_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [
                ((1, 3, 224, 224), (1, 3, 224, 1)),
                ((1, 3, 16, 1), (1, 1, 1, 16)),
            ]:
                data1 = torch.randn(shape1, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                data2 = torch.rand(shape2, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x = (20 * data1).to(data_type)
                y = (20 * data2).to(data_type) + 1
                out_cpu = x % y
                out_mlu = self.to_mlu_dtype(x, data_type) % self.to_mlu_dtype(
                    y, data_type
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_tensor_tensor_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [
                ((1, 3, 224, 224), (1, 3, 224, 1)),
                ((1, 3, 16, 1), (1, 1, 1, 16)),
            ]:
                data1 = torch.randn(shape1, dtype=torch.float)
                data2 = torch.rand(shape2, dtype=torch.float)
                x = (20 * data1).to(data_type)
                y = (20 * data2).to(data_type) + 1
                out_cpu = x[:, :, :, :1] % y[:, :, :1, :]
                out_mlu = (
                    self.to_mlu_dtype(x, data_type)[:, :, :, :1]
                    % self.to_mlu_dtype(y, data_type)[:, :, :1, :]
                )
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_tensor_scalar(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [
                (),
                (6, 0),
                (5),
                (7, 9),
                (9, 8, 7),
                (1, 2, 3, 4),
                (5, 6, 7, 8, 9),
                (9, 2, 12, 1, 2, 16),
            ]
            for shape in shape_list:
                x = (20 * torch.randn(shape, dtype=torch.float)).to(data_type)
                out_cpu = x % 8.0
                out_mlu = self.to_mlu_dtype(x, data_type) % 8.0
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_tensor_scalar_channels_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [(1, 2, 3, 4)]
            for shape in shape_list:
                data = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x = (20 * data).to(data_type) + 1
                out_cpu = x % 8.0
                out_mlu = self.to_mlu_dtype(x, data_type) % 8.0
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_tensor_scalar_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [(1, 2, 3, 4), (5, 6, 7, 8, 9), (9, 2, 12, 1, 2, 16)]
            for shape in shape_list:
                data = torch.randn(shape, dtype=torch.float)
                x = (20 * data).to(data_type)
                out_cpu = x[:, :, :2, :] % 8.0
                out_mlu = self.to_mlu_dtype(x, data_type)[:, :, :2, :] % 8.0
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_inplace_tensor_tensor(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [
                ((), ()),
                ((6), (6)),
                ((6, 0), (6, 0)),
                ((1, 3, 224, 224), (1, 3, 224, 1)),
                ((2, 2, 4, 2), (2)),
                ((2, 1, 2, 4), (1, 2, 4)),
            ]:
                x = (20 * torch.randn(shape1, dtype=torch.float)).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                y = (20 * torch.rand(shape2, dtype=torch.float)).to(data_type) + 1
                y_mlu = self.to_mlu_dtype(y, data_type)
                ptr1 = x_mlu.data_ptr()
                x.remainder_(y)
                x_mlu.remainder_(y_mlu)
                ptr2 = x_mlu.data_ptr()
                self.assertEqual(ptr1, ptr2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_inplace_tensor_tensor_channels_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1))]:
                data1 = torch.randn(shape1, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                data2 = torch.rand(shape2, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x = (20 * data1).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                y = (20 * data2).to(data_type) + 1
                y_mlu = self.to_mlu_dtype(y, data_type)
                ptr1 = x_mlu.data_ptr()
                x.remainder_(y)
                x_mlu.remainder_(y_mlu)
                ptr2 = x_mlu.data_ptr()
                self.assertEqual(ptr1, ptr2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_inplace_tensor_tensor_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1))]:
                data1 = torch.randn(shape1, dtype=torch.float)
                data2 = torch.rand(shape2, dtype=torch.float)
                x = (20 * data1).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                y = (20 * data2).to(data_type) + 1
                y_mlu = self.to_mlu_dtype(y, data_type)
                ptr1 = x_mlu.data_ptr()
                x[:, :, :, :1].remainder_(y[:, :, :1, :])
                x_mlu[:, :, :, :1].remainder_(y_mlu[:, :, :1, :])
                ptr2 = x_mlu.data_ptr()
                self.assertEqual(ptr1, ptr2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_inplace_tensor_scalar(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [
                (),
                (6, 0),
                (5),
                (7, 9),
                (9, 8, 7),
                (1, 2, 3, 4),
                (5, 6, 7, 8, 9),
                (9, 2, 12, 1, 2, 16),
            ]
            for shape in shape_list:
                x = (20 * torch.randn(shape, dtype=torch.float)).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                ptr1 = x_mlu.data_ptr()
                x.remainder_(8)
                x_mlu.remainder_(8)
                ptr2 = x_mlu.data_ptr()
                self.assertEqual(ptr1, ptr2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_inplace_tensor_scalar_channels_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [(1, 2, 3, 4)]
            for shape in shape_list:
                data = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x = (20 * data).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                ptr1 = x_mlu.data_ptr()
                x.remainder_(8)
                x_mlu.remainder_(8)
                ptr2 = x_mlu.data_ptr()
                self.assertEqual(ptr1, ptr2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_inplace_tensor_scalar_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [(9, 8, 7)]
            for shape in shape_list:
                data = torch.randn(shape, dtype=torch.float)
                x = (20 * data).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                ptr1 = x_mlu.data_ptr()
                x[:, :, :3].remainder_(8)
                x_mlu[:, :, :3].remainder_(8)
                ptr2 = x_mlu.data_ptr()
                self.assertEqual(ptr1, ptr2)
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_out_tensor_tensor(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [
                ((), ()),
                ((6), (6)),
                ((6, 0), (6, 0)),
                ((1, 3, 224, 224), (1, 3, 224, 1)),
                ((2, 2, 4, 2), (2)),
                ((1, 2), (2, 2, 4, 2)),
                ((2, 1, 2, 4), (1, 2, 4)),
                ((1, 3, 16, 1), (1, 1, 1, 16)),
            ]:
                x = (20 * torch.randn(shape1, dtype=torch.float)).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                y = (20 * torch.rand(shape2, dtype=torch.float)).to(data_type) + 1
                y_mlu = self.to_mlu_dtype(y, data_type)
                out_cpu = torch.zeros((1)).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros((1)).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, y, out=out_cpu)
                torch.remainder(x_mlu, y_mlu, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )
                broadcast_shape = torch._C._infer_size(x_mlu.shape, y_mlu.shape)
                out_cpu = torch.zeros(broadcast_shape).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros(broadcast_shape).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, y, out=out_cpu)
                torch.remainder(x_mlu, y_mlu, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_out_tensor_tensor_channels_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [
                ((1, 3, 224, 224), (1, 3, 224, 1)),
                ((1, 3, 16, 1), (1, 1, 1, 16)),
            ]:
                data1 = torch.randn(shape1, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                data2 = torch.rand(shape2, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x = (20 * data1).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                y = (20 * data2).to(data_type) + 1
                y_mlu = self.to_mlu_dtype(y, data_type)
                out_cpu = torch.zeros((1)).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros((1)).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, y, out=out_cpu)
                torch.remainder(x_mlu, y_mlu, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )
                broadcast_shape = torch._C._infer_size(x_mlu.shape, y_mlu.shape)
                out_cpu = torch.zeros(broadcast_shape).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros(broadcast_shape).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, y, out=out_cpu)
                torch.remainder(x_mlu, y_mlu, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_out_tensor_tensor_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            for shape1, shape2 in [((1, 3, 224, 224), (1, 3, 224, 1))]:
                data1 = torch.randn(shape1, dtype=torch.float)
                data2 = torch.rand(shape2, dtype=torch.float)
                x = (20 * data1).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                y = (20 * data2).to(data_type) + 1
                y_mlu = self.to_mlu_dtype(y, data_type)
                out_cpu = torch.zeros((1)).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros((1)).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x[:, :, :, :1], y[:, :, :1, :], out=out_cpu)
                torch.remainder(x_mlu[:, :, :, :1], y_mlu[:, :, :1, :], out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )
                broadcast_shape = torch._C._infer_size(x_mlu.shape, y_mlu.shape)
                out_cpu = torch.zeros(broadcast_shape).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros(broadcast_shape).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, y, out=out_cpu)
                torch.remainder(x_mlu, y_mlu, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_out_tensor_scalar(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [
                (),
                (5),
                (6, 0),
                (7, 9),
                (9, 8, 7),
                (1, 2, 3, 4),
                (5, 6, 7, 8, 9),
                (9, 2, 12, 1, 2, 16),
            ]
            for shape in shape_list:
                x = (20 * torch.rand(shape, dtype=torch.float)).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                out_cpu = torch.zeros((1)).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros((1)).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, 8, out=out_cpu)
                torch.remainder(x_mlu, 8, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )
                out_cpu = torch.zeros(shape).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros(shape).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, 8, out=out_cpu)
                torch.remainder(x_mlu, 8, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_out_tensor_scalar_channels_last(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [(1, 2, 3, 4)]
            for shape in shape_list:
                data = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x = (20 * data).to(data_type) + 1
                x_mlu = self.to_mlu_dtype(x, data_type)
                out_cpu = torch.zeros((1)).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros((1)).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, 8, out=out_cpu)
                torch.remainder(x_mlu, 8, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )
                out_cpu = torch.zeros(shape).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros(shape).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x, 8, out=out_cpu)
                torch.remainder(x_mlu, 8, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_out_tensor_scalar_not_dense(self):
        dtype_list = [(torch.float, 3e-3), (torch.int32, 0)]
        for data_type, err in dtype_list:
            shape_list = [(1, 2, 3, 4)]
            for shape in shape_list:
                data = torch.randn(shape, dtype=torch.float)
                x = (20 * data).to(data_type)
                x_mlu = self.to_mlu_dtype(x, data_type)
                out_cpu = torch.zeros((1)).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros((1)).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x[:, :, :, :3], 8, out=out_cpu)
                torch.remainder(x_mlu[:, :, :, :3], 8, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )
                out_cpu = torch.zeros(shape).to(data_type)
                out_cpu.resize_(0)
                out_mlu = torch.zeros(shape).to(data_type).to("mlu")
                out_mlu.resize_(0)
                torch.remainder(x[:, :, :, :3], 8, out=out_cpu)
                torch.remainder(x_mlu[:, :, :, :3], 8, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_mixed_memory_format(self):
        x = 20 * torch.randn((3, 1, 5, 1), dtype=torch.float).to(
            memory_format=torch.channels_last
        )
        y = 20 * torch.rand((4, 5, 1), dtype=torch.float)
        out_cpu = x % y
        out_mlu = self.to_mlu(x) % self.to_mlu(y)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

        y = 20 * torch.randn((3, 1, 5, 1), dtype=torch.float).to(
            memory_format=torch.channels_last
        )
        x = 20 * torch.rand((4, 5, 1), dtype=torch.float)
        out_cpu = x % y
        out_mlu = self.to_mlu(x) % self.to_mlu(y)
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_permute(self):
        shape_list = [
            (512, 1024, 2, 2, 4),
            (10, 3, 32, 32),
            (2, 3, 4),
            (254, 254, 112, 1, 1, 3),
        ]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1), (0, 5, 4, 3, 2, 1)]
        for i in range(4):
            x = torch.rand(shape_list[i], dtype=torch.float) * 10
            y = torch.rand(shape_list[i], dtype=torch.float) * 10 + 1
            out = torch.randn(shape_list[i], dtype=torch.float)
            x_mlu = copy.deepcopy(x).to("mlu")
            y_mlu = copy.deepcopy(y).to("mlu")
            out_mlu = copy.deepcopy(out).to("mlu")
            x, y, out = (
                x.permute(permute_shape[i]),
                y.permute(permute_shape[i]),
                out.permute(permute_shape[i]),
            )
            x_mlu, y_mlu, out_mlu = (
                x_mlu.permute(permute_shape[i]),
                y_mlu.permute(permute_shape[i]),
                out_mlu.permute(permute_shape[i]),
            )
            out_cpu = x % y
            out_mlu = x_mlu % y_mlu
            self.assertTrue(out_cpu.stride() == out_mlu.stride())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 3e-3, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_remainder_exception(self):
        x_mlu = (torch.rand(1, 16) * 10).to("mlu")
        y_mlu = (torch.rand(16, 1) * 10).to("mlu")
        ref_msg = (
            r"^output with shape \[1, 16\] doesn't match the broadcast shape \[16, 16\]"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            x_mlu.remainder_(y_mlu)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_remainder_bfloat16(self):
        parameter_list = [
            (torch.bfloat16, 3e-3, False),
            (torch.bfloat16, 3e-3, True),
        ]
        shape_list = [
            ((), ()),
            ((6, 0, 3), (6, 0, 1)),
            ((1, 3, 224, 224), (1, 3, 224, 1)),
            ((2, 2, 4, 2), (2)),
        ]
        for parameter, shape in product(parameter_list, shape_list):
            data_type, err, other_is_scalar = parameter
            shape1, shape2 = shape
            x = (20 * torch.randn(shape1, dtype=torch.float)).to(data_type)
            y = (
                (20 * torch.rand(shape2, dtype=torch.float)).to(data_type) + 1
                if other_is_scalar is False
                else 2.0
            )
            x_cpu = torch.nn.Parameter(x)
            y_cpu = y
            x_mlu = torch.nn.Parameter(x.mlu())
            y_mlu = y.mlu() if other_is_scalar is False else y
            out_cpu = x_cpu % y_cpu
            out_mlu = x_mlu % y_mlu
            grad = torch.randn(out_cpu.shape)
            grad_mlu = grad.mlu()
            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )
            self.assertTensorsEqual(
                x_cpu.grad.float(), x_mlu.grad.cpu().float(), err, use_MSE=True
            )


if __name__ == "__main__":
    unittest.main()
