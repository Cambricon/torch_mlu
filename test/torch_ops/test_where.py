from __future__ import print_function

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
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestWhereOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_where(self):
        con_shape_list = [
            (343, 4),
            (80, 4),
            (1, 4, 5),
            (1, 4, 5, 6),
            (),
            (0),
            (1),
            (343, 4),
        ]
        x_shape_list = [(343, 4), (80, 4), (3, 4, 5), (1, 4, 5, 6), (), (0), (1), (1)]
        y_shape_list = [(343, 4), (80, 4), (3, 4, 5), (3, 4, 5, 6), (), (0), (1), (1)]
        dtype = [
            torch.float32,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.long,
            torch.double,
        ]
        cond_dtypes = [torch.bool, torch.uint8]
        for con_shape, x_shape, y_shape in zip(
            con_shape_list, x_shape_list, y_shape_list
        ):
            for xy_dtype in dtype:
                for cond_dtype in cond_dtypes:
                    condition = torch.randn(con_shape).to(dtype=cond_dtype)
                    x = torch.randn(x_shape).to(dtype=xy_dtype)
                    y = torch.randn(y_shape).to(dtype=xy_dtype)
                    out_cpu = torch.where(condition, x, y)
                    condition_mlu = condition.to("mlu")
                    x_mlu = x.to("mlu")
                    y_mlu = y.to("mlu")
                    out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_where_PYTORCH_11152(self):
        dtype = [torch.float32]
        cond_dtypes = [torch.bool]
        for xy_dtype in dtype:
            for cond_dtype in cond_dtypes:
                condition = torch.randn(1, 4, 1, 64, 64).to(dtype=cond_dtype)
                x = torch.randn(1, 4, 1, 64, 64).to(dtype=xy_dtype)
                y = torch.randn(1, 4, 1, 64, 64).to(dtype=xy_dtype)
                x.as_strided_(x.size(), stride=(4, 1, 4, 256, 4))
                y.as_strided_(y.size(), stride=(16384, 1, 16384, 256, 4))
                condition.as_strided_(
                    condition.size(), stride=(16384, 1, 16384, 256, 4)
                )
                out_cpu = torch.where(condition, x, y)
                condition_mlu = condition.to("mlu")
                x_mlu = x.to("mlu")
                y_mlu = y.to("mlu")
                out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_where_bfloat16(self):
        con_shape_list = [
            (343, 4),
            (80, 4),
            (1, 4, 5),
            (1, 4, 5, 6),
            (),
            (0),
            (1),
            (343, 4),
        ]
        x_shape_list = [(343, 4), (80, 4), (3, 4, 5), (1, 4, 5, 6), (), (0), (1), (1)]
        y_shape_list = [(343, 4), (80, 4), (3, 4, 5), (3, 4, 5, 6), (), (0), (1), (1)]
        dtype = [torch.bfloat16]
        cond_dtypes = [torch.bool, torch.uint8]
        for con_shape, x_shape, y_shape in zip(
            con_shape_list, x_shape_list, y_shape_list
        ):
            for xy_dtype in dtype:
                for cond_dtype in cond_dtypes:
                    condition = torch.randn(con_shape).to(dtype=cond_dtype)
                    x = torch.randn(x_shape).to(dtype=xy_dtype)
                    y = torch.randn(y_shape).to(dtype=xy_dtype)
                    out_cpu = torch.where(condition, x, y)
                    condition_mlu = condition.to("mlu")
                    x_mlu = x.to("mlu")
                    y_mlu = y.to("mlu")
                    out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_where_condition_tensor(self):
        shape = [(343, 4), (80, 4), (1, 4, 5), (1, 4, 5, 6), ()]
        dtype = [
            torch.bool,
            torch.float32,
            torch.int32,
            torch.double,
            torch.int64,
            torch.long,
        ]
        for cond_shape, cond_dtype in zip(shape, dtype):
            condition = torch.randn(cond_shape).to(dtype=cond_dtype)
            result_cpu = torch.where(condition)
            result_mlu = torch.where(self.to_device(condition))
            self.assertEqual(len(result_mlu), len(result_cpu))
            for out_cpu, out_mlu in zip(result_cpu, result_mlu):
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)
        for cond_shape, cond_dtype in zip(shape, dtype):
            condition = torch.ones(cond_shape).to(dtype=cond_dtype)
            result_cpu = torch.where(condition)
            result_mlu = torch.where(self.to_device(condition))
            self.assertEqual(len(result_mlu), len(result_cpu))
            for out_cpu, out_mlu in zip(result_cpu, result_mlu):
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_where_backward(self):
        con_shape_list = [(343, 4), (80, 4), (1, 4, 5), (1, 4, 5, 6), ()]
        x_shape_list = [(343, 4), (80, 4), (3, 4, 5), (1, 4, 5, 6), ()]
        y_shape_list = [(343, 4), (80, 4), (3, 4, 5), (3, 4, 5, 6), ()]
        for con_shape, x_shape, y_shape in zip(
            con_shape_list, x_shape_list, y_shape_list
        ):
            condition = torch.randn(con_shape, dtype=torch.float, requires_grad=True)
            x = torch.randn(x_shape, dtype=torch.float, requires_grad=True)
            y = torch.randn(y_shape, dtype=torch.float, requires_grad=True)
            out_cpu = torch.where(x > condition, x, y)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_x_cpu = copy.deepcopy(x.grad)
            grad_y_cpu = copy.deepcopy(y.grad)
            x.grad.zero_()
            y.grad.zero_()
            condition_mlu = condition.to("mlu")
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = torch.where(x_mlu > condition_mlu, x_mlu, y_mlu)
            out_mlu.backward(self.to_mlu(grad.to("mlu")))
            grad_x_mlu = copy.deepcopy(x.grad)
            grad_y_mlu = copy.deepcopy(y.grad)
            self.assertTensorsEqual(grad_x_cpu, grad_x_mlu, 0.003, use_MSE=True)
            self.assertTensorsEqual(grad_y_cpu, grad_y_mlu, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_where_dtype_exception(self):
        c = torch.tensor(1).to("mlu")
        a = torch.tensor(1).to("mlu")
        b = torch.randn(1).to("mlu")
        ref_msg = "where expected condition to be a boolean tensor, but got a tensor with dtype"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.where(c, a, b)

        c = torch.randint(2, (2, 3)).to(torch.bool).to("mlu")
        a = torch.randn((2, 3)).to(torch.uint8).to("mlu")
        b = torch.randn((2, 3)).to(torch.uint8).to("mlu")
        ref_msg = "\"cnnl_where\" not implemented for 'Byte'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.where(c, a, b)

        c = torch.randint(2, (2, 3)).to(torch.bool).to("mlu")
        a = torch.randn((2, 3)).to(torch.bool).to("mlu")
        b = torch.randn((2, 3)).to(torch.bool).to("mlu")
        ref_msg = "\"cnnl_where\" not implemented for 'Bool'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.where(c, a, b)

    # @unittest.skip("not test")
    @testinfo()
    def test_where_channels_last(self):
        con_shape_list = [
            (1, 4, 5, 6),
        ]
        x_shape_list = [
            (1, 4, 5, 6),
        ]
        y_shape_list = [
            (3, 4, 5, 6),
        ]
        dtype = [
            torch.float32,
        ]
        for con_shape, x_shape, y_shape, xy_dtype in zip(
            con_shape_list, x_shape_list, y_shape_list, dtype
        ):
            condition = torch.randint(2, con_shape).to(dtype=torch.bool)
            condition = condition.to(memory_format=torch.channels_last)
            x = torch.randn(x_shape, dtype=xy_dtype).to(
                memory_format=torch.channels_last
            )
            y = torch.randn(y_shape, dtype=xy_dtype).to(
                memory_format=torch.channels_last
            )
            out_cpu = torch.where(condition, x, y)
            condition_mlu = condition.to("mlu")
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    def test_where_one_dim_channels_last(self):
        con_shape_list = [
            (4, 1, 5, 6),
        ]
        x_shape_list = [
            (4, 1, 5, 6),
        ]
        y_shape_list = [
            (4, 1, 5, 6),
        ]
        dtype = [
            torch.float32,
        ]
        for con_shape, x_shape, y_shape, xy_dtype in zip(
            con_shape_list, x_shape_list, y_shape_list, dtype
        ):
            condition = torch.randint(2, con_shape).to(dtype=torch.bool)
            condition = condition.to(memory_format=torch.channels_last)
            x = torch.randn(x_shape, dtype=xy_dtype).to(
                memory_format=torch.channels_last
            )
            y = torch.randn(y_shape, dtype=xy_dtype).to(
                memory_format=torch.channels_last
            )
            out_cpu = torch.where(condition, x, y)
            condition_mlu = condition.to("mlu")
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    def test_where_scalar_channels_last_contiguous(self):
        con_shape_list = [
            (2, 1, 2, 2),
        ]
        x_shape_list = [
            (1, 3, 1, 1),
        ]
        dtype = [
            torch.float32,
        ]
        for con_shape, x_shape, xy_dtype in zip(con_shape_list, x_shape_list, dtype):
            condition = torch.randint(2, con_shape).to(dtype=torch.bool)
            condition = condition.to(memory_format=torch.channels_last)
            x = torch.randn(x_shape, dtype=xy_dtype).to(
                memory_format=torch.channels_last
            )
            y = torch.tensor(1)
            out_cpu = torch.where(condition, x, y)
            condition_mlu = condition.to("mlu")
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_where_not_dense(self):
        con_shape_list = [
            (1, 4, 5, 6),
        ]
        x_shape_list = [
            (1, 4, 5, 6),
        ]
        y_shape_list = [
            (3, 4, 5, 6),
        ]
        dtype = [
            torch.float32,
        ]
        for con_shape, x_shape, y_shape, xy_dtype in zip(
            con_shape_list, x_shape_list, y_shape_list, dtype
        ):
            condition = torch.randint(2, con_shape).to(dtype=torch.bool)
            x = torch.randn(x_shape).to(dtype=xy_dtype)
            y = torch.randn(y_shape).to(dtype=xy_dtype)
            out_cpu = torch.where(condition[..., :3], x[..., :3], y[..., :3])
            condition_mlu = condition.to("mlu")[..., :3]
            x_mlu = x.to("mlu")[..., :3]
            y_mlu = y.to("mlu")[..., :3]
            out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_where_not_dense_channels_last(self):
        con_shape_list = [
            (1, 4, 5, 6),
        ]
        x_shape_list = [
            (1, 4, 5, 6),
        ]
        y_shape_list = [
            (3, 4, 5, 6),
        ]
        dtype = [
            torch.float32,
        ]
        for con_shape, x_shape, y_shape, xy_dtype in zip(
            con_shape_list, x_shape_list, y_shape_list, dtype
        ):
            condition = torch.randint(2, con_shape).to(
                dtype=torch.bool, memory_format=torch.channels_last
            )[..., :3]
            x = (
                torch.randn(x_shape)
                .to(dtype=xy_dtype)
                .to(memory_format=torch.channels_last)[..., :3]
            )
            y = (
                torch.randn(y_shape)
                .to(dtype=xy_dtype)
                .to(memory_format=torch.channels_last)[..., :3]
            )
            out_cpu = torch.where(condition, x, y)
            condition_mlu = condition.to("mlu")
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    def test_where_anomalistic_stride(self):
        con_shape_list = [
            (1, 128, 1, 64),
        ]
        x_shape_list = [
            (1, 128, 1, 64),
        ]
        y_shape_list = [
            (1, 128, 1, 64),
        ]
        dtype = [
            torch.float32,
        ]
        for con_shape, x_shape, y_shape, xy_dtype in zip(
            con_shape_list, x_shape_list, y_shape_list, dtype
        ):
            condition = torch.randint(2, con_shape).to(dtype=torch.bool)
            condition = condition.to(memory_format=torch.channels_last)
            x = torch.randn(x_shape, dtype=xy_dtype)
            # set the first stride to a wrong number of CL format.
            x = torch.as_strided(x, (1, 128, 1, 64), (128, 1, 8192, 128))
            y = torch.randn(y_shape, dtype=xy_dtype).to(
                memory_format=torch.channels_last
            )
            out_cpu = torch.where(condition, x, y)
            condition_mlu = condition.to("mlu")
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_where_large(self):
        con_shape_list = [(5, 1024, 1024, 1024), (4294967296, 1)]
        x_shape_list = [(5, 1024, 1024, 1024), (4294967296, 1)]
        y_shape_list = [(5, 1024, 1024, 1024), (4294967296, 1)]
        dtype = [torch.int8]
        cond_dtypes = [torch.bool, torch.uint8]
        for con_shape, x_shape, y_shape, xy_dtype, cond_dtype in zip(
            con_shape_list, x_shape_list, y_shape_list, dtype, cond_dtypes
        ):
            condition = torch.randint(2, con_shape).to(dtype=cond_dtype)
            x = torch.randn(x_shape).to(dtype=xy_dtype)
            y = torch.randn(y_shape).to(dtype=xy_dtype)
            out_cpu = torch.where(condition, x, y)
            condition_mlu = condition.to("mlu")
            x_mlu = x.to("mlu")
            y_mlu = y.to("mlu")
            out_mlu = torch.where(condition_mlu, x_mlu, y_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
