from __future__ import print_function

import logging
import os
import sys

import copy
import random
import unittest

import torch
from torch.testing._internal.common_utils import make_tensor

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestNantonumOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_contiguous(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf)
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_channel_last(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x = self.convert_to_channel_last(x)
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf)
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_not_dense(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf)
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_out_contiguous(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            y = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            y_mlu = copy.deepcopy(y).to("mlu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf, out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_out_channel_last(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            y = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            y_mlu = copy.deepcopy(y).to("mlu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x = self.convert_to_channel_last(x)
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf, out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_out_not_dense(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            y = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            y_mlu = copy.deepcopy(y).to("mlu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf, out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)
        for dtype in dtype_list:
            out_cpu = make_tensor(
                [8, 8], low=0.0, high=100.0, dtype=dtype, device="cpu"
            )
            out_mlu = self.to_mlu(out_cpu)
            x = make_tensor([4, 4], low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, 3, (3,)), extremals):
                    x[idx] = extremal
            x_mlu = self.to_mlu(x)
            torch.nan_to_num(x, out=out_cpu[:2, :2])
            torch.nan_to_num(x_mlu, out=out_mlu[:2, :2])
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_out_shape_contiguous(self):
        dtype_list = [torch.float, torch.half, torch.bool]
        for dtype in dtype_list:
            x = make_tensor([10000], low=0.0, high=100.0, dtype=dtype, device="cpu")
            y = make_tensor([1000], low=0.0, high=100.0, dtype=dtype, device="cpu")
            y_mlu = copy.deepcopy(y).to("mlu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, 3, (3,)), extremals):
                    x[idx] = extremal
            x_mlu = self.to_mlu(x)
            out_cpu = torch.nan_to_num(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.nan_to_num(x_mlu, out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            assert ori_ptr != out_ptr
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

        for dtype in dtype_list:
            x = make_tensor([1000], low=0.0, high=100.0, dtype=dtype, device="cpu")
            y = make_tensor([10000], low=0.0, high=100.0, dtype=dtype, device="cpu")
            y_mlu = copy.deepcopy(y).to("mlu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, 3, (3,)), extremals):
                    x[idx] = extremal
            x_mlu = self.to_mlu(x)
            out_cpu = torch.nan_to_num(x, out=y)
            ori_ptr = y_mlu.data_ptr()
            out_mlu = torch.nan_to_num(self.to_mlu(x), out=y_mlu)
            out_ptr = y_mlu.data_ptr()
            self.assertEqual(ori_ptr, out_ptr)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_t_contiguous(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = x.nan_to_num(nan, posinf, neginf)
            out_mlu = x_mlu.nan_to_num(nan, posinf, neginf)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_t_channel_last(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x = self.convert_to_channel_last(x)
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = x.nan_to_num(nan, posinf, neginf)
            out_mlu = x_mlu.nan_to_num(nan, posinf, neginf)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_t_not_dense(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = x.nan_to_num(nan, posinf, neginf)
            out_mlu = x_mlu.nan_to_num(nan, posinf, neginf)
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_inplace_contiguous(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            input_ptr = x_mlu.data_ptr()
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            x.nan_to_num_(nan, posinf, neginf)
            x_mlu.nan_to_num_(nan, posinf, neginf)
            self.assertEqual(input_ptr, x_mlu.data_ptr())
            if dtype == torch.half:
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_inplace_channel_last(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x = self.convert_to_channel_last(x)
            x_mlu = x.to("mlu")
            input_ptr = x_mlu.data_ptr()
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            x.nan_to_num_(nan, posinf, neginf)
            x_mlu.nan_to_num_(nan, posinf, neginf)
            self.assertEqual(input_ptr, x_mlu.data_ptr())
            if dtype == torch.half:
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_inplace_not_dense(self):
        shape_list = [
            (4, 1, 6, 40000, 1),
            (10, 3, 32, 32),
            (7, 3, 4),
            (64, 64),
            (10),
            (1),
        ]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            if len(shape) == 4:
                x = x[:, :, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, :, : int(shape[-1] / 2)]
            elif len(shape) == 3:
                x = x[:, :, : int(shape[-1] / 2)]
                x_mlu = x_mlu[:, :, : int(shape[-1] / 2)]
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            input_ptr = x_mlu.data_ptr()
            x.nan_to_num_(nan, posinf, neginf)
            x_mlu.nan_to_num_(nan, posinf, neginf)
            self.assertEqual(input_ptr, x_mlu.data_ptr())
            if dtype == torch.half:
                self.assertTensorsEqual(
                    x.float(), x_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(x, x_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_permute(self):
        shape_list = [(4, 1, 6, 40000, 1), (10, 3, 32, 32), (7, 3, 4)]
        permute_shape = [(3, 2, 1, 4, 0), (0, 3, 2, 1), (2, 0, 1)]
        dtype_list = [torch.float, torch.half, torch.bool]
        for shape, p_shape, dtype in zip(shape_list, permute_shape, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            out = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            out_mlu = out.to("mlu")
            x, out = x.permute(p_shape), out.permute(p_shape)
            x_mlu, out_mlu = x_mlu.permute(p_shape), out_mlu.permute(p_shape)
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            torch.nan_to_num(x, nan, posinf, neginf, out=out)
            torch.nan_to_num(x_mlu, nan, posinf, neginf, out=out_mlu)
            self.assertTrue(out.stride() == out_mlu.stride())
            self.assertTrue(out.storage_offset() == out_mlu.storage_offset())
            if dtype == torch.half:
                self.assertTensorsEqual(
                    out.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
                )
            else:
                self.assertTensorsEqual(out, out_mlu.cpu(), 0.0, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nan_to_num_exception(self):
        x_mlu = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14]).to(
            "mlu"
        )
        y_out = make_tensor((1, 4), low=0.0, high=100.0, dtype=torch.half, device="mlu")
        ref_msg = "nan_to_num: dtype of out: Half should be same as input: Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.nan_to_num(x_mlu, out=y_out)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    def test_nan_to_num_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        dtype_list = [torch.half]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf)
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_nan_to_num_bfloat16(self):
        shape_list = [
            (10, 3, 32, 32),
        ]
        dtype_list = [
            torch.bfloat16,
        ]
        for shape, dtype in zip(shape_list, dtype_list):
            x = make_tensor(shape, low=0.0, high=100.0, dtype=dtype, device="cpu")
            if dtype.is_floating_point:
                extremals = [float("nan"), float("inf"), -float("inf")]
                for idx, extremal in zip(torch.randint(0, shape[0], (3,)), extremals):
                    x[idx] = extremal
            x_mlu = x.to("mlu")
            nan = random.random() if random.random() > 0.5 else None
            posinf = random.random() * 5 if random.random() > 0.5 else None
            neginf = random.random() * 10 if random.random() > 0.5 else None
            out_cpu = torch.nan_to_num(x, nan, posinf, neginf)
            out_mlu = torch.nan_to_num(x_mlu, nan, posinf, neginf)
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), 0.0, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
