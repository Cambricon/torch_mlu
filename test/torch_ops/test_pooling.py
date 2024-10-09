from __future__ import print_function

import sys
import os
import unittest
import logging
from itertools import product
from pandas._libs.tslibs import dtypes

import torch
from torch import nn
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    read_card_info,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413 C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestPoolingOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool1d_non_batch(self):
        shape_list = [(7, 7), (6, 12), (13, 16), (16, 7), (6, 8), (23, 13)]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtypes,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, t in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            avg_pool = nn.AvgPool1d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # not dense
            output_cpu = avg_pool(input_cpu[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool1d_batch(self):
        shape_list = [(16, 7, 7), (6, 8, 16), (23, 13, 64), (0, 128, 128)]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtypes,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, t in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            avg_pool = nn.AvgPool1d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # not dense
            output_cpu = avg_pool(input_cpu[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool2d_non_batch(self):
        shape_list = [
            (6, 7, 7),
            (3, 6, 12),
            (4, 13, 16),
            (16, 7, 7),
            (6, 8, 16),
            (23, 13, 64),
        ]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtypes,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, t in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            avg_pool = nn.AvgPool2d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # not dense
            output_cpu = avg_pool(input_cpu[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool2d_batch(self):
        shape_list = [(8, 16, 7, 7), (16, 6, 8, 16), (4, 23, 13, 64), (0, 2, 128, 128)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtypes,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            include_pad,
            t,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=t).to(memory_format=memory_format)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            avg_pool = nn.AvgPool2d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # not dense
            output_cpu = avg_pool(input_cpu[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool1d_non_batch(self):
        in_shapes = [(12, 12), (128, 128)]
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            kernel_v,
            stride_v,
            padding_v,
            return_indices_v,
            ceil_mode_v,
            dtypes,
            in_shapes,
        ]
        for kernel, stride, padding, return_indices, ceil_mode, t, in_shape in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            output_cpu = F.max_pool1d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool1d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

            # not dense
            output_cpu = F.max_pool1d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool1d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool1d_index(self):
        # check index value since cnnl return local index.
        input_dtypes = [torch.half, torch.float, torch.double]
        for input_dtype in input_dtypes:
            demo_input = torch.Tensor([[1, 3, 4, 2], [2, 5, 6, 7], [3, 8, 9, 6]]).to(
                input_dtype
            )
            out_put_gt = torch.Tensor([[1, 2, 2], [1, 2, 3], [1, 2, 2]]).to(torch.long)
            out_put_gt_mlu = torch.Tensor([[1, 1, 0], [1, 1, 1], [1, 1, 0]]).to(
                torch.long
            )
            output_demo_cpu = F.max_pool1d(
                demo_input.float(),
                kernel_size=2,
                stride=1,
                padding=0,
                dilation=1,
                return_indices=True,
                ceil_mode=False,
            )
            output_demo = F.max_pool1d(
                self.to_mlu(demo_input),
                kernel_size=2,
                stride=1,
                padding=0,
                dilation=1,
                return_indices=True,
                ceil_mode=False,
            )
            self.assertEqual(output_demo_cpu[1], out_put_gt)
            self.assertEqual(output_demo[1].cpu(), out_put_gt_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool2d_non_batch(self):
        in_shapes = [(2, 12, 12), (4, 128, 128)]
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            kernel_v,
            stride_v,
            padding_v,
            return_indices_v,
            ceil_mode_v,
            dtypes,
            in_shapes,
        ]
        for kernel, stride, padding, return_indices, ceil_mode, t, in_shape in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            output_cpu = F.max_pool2d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool2d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

            # not dense
            output_cpu = F.max_pool2d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool2d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool2d_index(self):
        # check index value since cnnl return local index.
        input_dtypes = [torch.half, torch.float, torch.double]
        for input_dtype in input_dtypes:
            demo_input = torch.Tensor(
                [[[[1, 3, 4, 2], [2, 5, 6, 7], [3, 8, 9, 6]]]]
            ).to(input_dtype)
            out_put_gt = torch.Tensor(
                [
                    [
                        [
                            [5, 6, 7],
                            [9, 10, 10],
                        ]
                    ]
                ]
            ).to(torch.long)
            out_put_gt_mlu = torch.Tensor(
                [
                    [
                        [
                            [3, 3, 3],
                            [3, 3, 2],
                        ]
                    ]
                ]
            ).to(torch.long)
            output_demo_cpu = F.max_pool2d(
                demo_input.float(),
                kernel_size=2,
                stride=1,
                padding=0,
                dilation=1,
                return_indices=True,
                ceil_mode=False,
            )
            output_demo = F.max_pool2d(
                self.to_mlu(demo_input),
                kernel_size=2,
                stride=1,
                padding=0,
                dilation=1,
                return_indices=True,
                ceil_mode=False,
            )
            self.assertEqual(output_demo_cpu[1], out_put_gt)
            self.assertEqual(output_demo[1].cpu(), out_put_gt_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool1d_batch(self):
        in_shapes = [(2, 12, 12), (0, 128, 128)]
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            kernel_v,
            stride_v,
            padding_v,
            return_indices_v,
            ceil_mode_v,
            dtypes,
            in_shapes,
        ]
        for kernel, stride, padding, return_indices, ceil_mode, t, in_shape in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            output_cpu = F.max_pool1d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool1d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

            # not dense
            output_cpu = F.max_pool1d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool1d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool2d_batch(self):
        in_shapes = [(4, 2, 12, 12), (0, 2, 128, 128)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            memory_format_list,
            kernel_v,
            stride_v,
            padding_v,
            return_indices_v,
            ceil_mode_v,
            dtypes,
            in_shapes,
        ]
        for (
            memory_format,
            kernel,
            stride,
            padding,
            return_indices,
            ceil_mode,
            t,
            in_shape,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=t).to(memory_format=memory_format)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            output_cpu = F.max_pool2d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool2d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

            # not dense
            output_cpu = F.max_pool2d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool2d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu(), 3e-3, use_MSE=True
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool3d_non_batch(self):
        shape_list = [(2048, 1, 7, 7), (192, 8, 28, 28)]
        kernel_v = [(1, 7, 7), (1, 3, 3)]
        stride_v = [(1, 1, 1), (1, 1, 1)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        include_pad_v = [True, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtypes,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, t in zip(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            # test nn module
            avg_pool = nn.AvgPool3d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test nn module not dense
            output_cpu = avg_pool(input_cpu[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test function
            output_cpu = F.avg_pool3d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_mlu = F.avg_pool3d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test function not dense
            output_cpu = F.avg_pool3d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_mlu = F.avg_pool3d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool3d_batch(self):
        shape_list = [(12, 2048, 1, 7, 7), (12, 192, 8, 28, 28)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(1, 7, 7), (1, 3, 3)]
        stride_v = [(1, 1, 1), (1, 1, 1)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        include_pad_v = [True, True]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtypes,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            include_pad,
            t,
        ) in zip(*loop_var):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            # test nn module
            avg_pool = nn.AvgPool3d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test nn module not dense
            output_cpu = avg_pool(input_cpu[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test function
            output_cpu = F.avg_pool3d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_mlu = F.avg_pool3d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test function not dense
            output_cpu = F.avg_pool3d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_mlu = F.avg_pool3d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool3d_non_batch(self):
        shape_list = [(2048, 2, 7, 7), (128, 8, 112, 112), (0, 128, 8, 8)]
        kernel_v = [(2, 1, 1), (2, 3, 3)]
        stride_v = [(1, 1, 1), (2, 2, 2)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        return_indices_v = [True, False]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            dtypes,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, return_indices, t in zip(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            # test nn module
            max_pool = nn.MaxPool3d(
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                ceil_mode=ceil_mode,
                return_indices=return_indices,
            )
            output_cpu = max_pool(input_cpu)
            output_mlu = max_pool(self.to_mlu(input_t))
            output_cpu_not_dense = max_pool(input_cpu[:2, ...])
            output_mlu_not_dense = max_pool(self.to_mlu(input_t)[:2, ...])
            if return_indices is True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense[0],
                    output_mlu_not_dense[0].cpu().float(),
                    3e-3,
                    use_MSE=True,
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense,
                    output_mlu_not_dense.cpu().float(),
                    3e-3,
                    use_MSE=True,
                )

            # test function
            output_cpu = F.max_pool3d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            output_mlu = F.max_pool3d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            output_cpu_not_dense = F.max_pool3d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            output_mlu_not_dense = F.max_pool3d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices is True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense[0],
                    output_mlu_not_dense[0].cpu().float(),
                    3e-3,
                    use_MSE=True,
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
                self.assertIs(output_mlu_not_dense[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense,
                    output_mlu_not_dense.cpu().float(),
                    3e-3,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool3d_index(self):
        # check index value since cnnl return local index.
        input_dtypes = [torch.half, torch.float, torch.double]
        for input_dtype in input_dtypes:
            demo_input = torch.Tensor(
                [
                    [
                        [
                            [[1, 3, 4, 2], [2, 5, 6, 7], [3, 8, 9, 6]],
                            [[1, 3, 4, 2], [2, 5, 6, 7], [3, 8, 9, 6]],
                        ]
                    ]
                ]
            ).to(input_dtype)
            out_put_gt = torch.Tensor([[[[[5, 6, 7], [9, 10, 10]]]]]).to(torch.long)
            out_put_gt_mlu = torch.Tensor([[[[[3, 3, 3], [3, 3, 2]]]]]).to(torch.long)
            output_demo_cpu = F.max_pool3d(
                demo_input.float(),
                kernel_size=2,
                stride=1,
                padding=0,
                dilation=1,
                return_indices=True,
                ceil_mode=False,
            )
            output_demo = F.max_pool3d(
                self.to_mlu(demo_input),
                kernel_size=2,
                stride=1,
                padding=0,
                dilation=1,
                return_indices=True,
                ceil_mode=False,
            )
            self.assertEqual(output_demo_cpu[1], out_put_gt)
            self.assertEqual(output_demo[1].cpu(), out_put_gt_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool3d_batch(self):
        shape_list = [(12, 2048, 2, 7, 7), (12, 128, 8, 112, 112), (0, 12, 8, 128, 128)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(2, 1, 1), (2, 3, 3)]
        stride_v = [(1, 1, 1), (2, 2, 2)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        return_indices_v = [True, False]
        dtypes = [torch.float, torch.half, torch.double]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            dtypes,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            t,
        ) in zip(*loop_var):
            input_t = torch.randn(in_shape, dtype=t)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            # test nn module
            max_pool = nn.MaxPool3d(
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                ceil_mode=ceil_mode,
                return_indices=return_indices,
            )
            output_cpu = max_pool(input_cpu)
            output_mlu = max_pool(self.to_mlu(input_t))
            output_cpu_not_dense = max_pool(input_cpu[:2, ...])
            output_mlu_not_dense = max_pool(self.to_mlu(input_t)[:2, ...])
            if return_indices is True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense[0],
                    output_mlu_not_dense[0].cpu().float(),
                    3e-3,
                    use_MSE=True,
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense,
                    output_mlu_not_dense.cpu().float(),
                    3e-3,
                    use_MSE=True,
                )

            # test function
            output_cpu = F.max_pool3d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            output_mlu = F.max_pool3d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            output_cpu_not_dense = F.max_pool3d(
                input_cpu[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            output_mlu_not_dense = F.max_pool3d(
                self.to_mlu(input_t)[:2, ...],
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices is True:
                self.assertTensorsEqual(
                    output_cpu[0], output_mlu[0].cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense[0],
                    output_mlu_not_dense[0].cpu().float(),
                    3e-3,
                    use_MSE=True,
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
                self.assertIs(output_mlu_not_dense[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
                self.assertTensorsEqual(
                    output_cpu_not_dense,
                    output_mlu_not_dense.cpu().float(),
                    3e-3,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool2d_exception(self):
        input = torch.randn((2, 3, 8, 8), dtype=torch.float).to("mlu")
        m = nn.MaxPool2d(kernel_size=(3, 3, 3), stride=2)
        m = m.to("mlu")
        ref_msg = r"^max_pool2d: kernel_size must either be a single int,"
        ref_msg = ref_msg + r" or a tuple of two ints$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2, 3, 8, 8), dtype=torch.float).to("mlu")
        m = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2, 2))
        m = m.to("mlu")
        ref_msg = r"^max_pool2d: stride must either be omitted,"
        ref_msg = ref_msg + r" a single int, or a tuple of two ints$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2, 3, 8, 8), dtype=torch.float).to("mlu")
        m = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 2))
        m = m.to("mlu")
        ref_msg = r"^max_pool2d: dilation must be either a single int, or a tuple of two ints,"
        ref_msg = ref_msg + r" and cnnl pool2d only supports defalut dilation value$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2, 512, 512, 512), dtype=torch.float).to("mlu")
        m = nn.MaxPool2d(kernel_size=(400, 400), stride=(2, 2))
        m = m.to("mlu")
        ref_msg = r"^max_pool2d: The kernel size should be"
        ref_msg = ref_msg + r" smaller than 65535, while this kernel size is 160000"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((64, 64, 3, 0), dtype=torch.float).to("mlu")
        m = nn.MaxPool2d(
            kernel_size=(8, 8),
            stride=(2, 2),
            padding=0,
            return_indices=True,
            dilation=1,
            ceil_mode=False,
        )
        m = m.to("mlu")
        ref_msg = r"Expected 3D or 4D \(batch mode\) tensor with optional 0 dim "
        ref_msg = ref_msg + r"batch size for input, but got:\[64, 64, 3, 0\]$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool2d_exception(self):
        input = torch.randn((2, 3, 8, 8), dtype=torch.float).to("mlu")
        m = nn.AvgPool2d(kernel_size=(3, 3), stride=2, divisor_override=3)
        ref_msg = r"^divisor_override is not supported$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((64, 64, 3, 0), dtype=torch.float).to("mlu")
        m = nn.AvgPool2d(kernel_size=(3, 3), stride=2)
        ref_msg = r"Expected 3D or 4D \(batch mode\) tensor with optional 0 dim "
        ref_msg = ref_msg + r"batch size for input, but got:\[64, 64, 3, 0\]$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool3d_exception(self):
        input = torch.randn((2, 3, 4, 4, 4), dtype=torch.float).to("mlu")
        m = torch.nn.AvgPool3d((2, 2, 2), divisor_override=3)
        ref_msg = r"^divisor_override is not supported$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((0, 3, 4, 4, 4, 5), dtype=torch.float).to("mlu")
        m = torch.nn.AvgPool3d((2, 2, 2))
        ref_msg = r"non-empty 4D or 5D \(batch mode\) tensor expected for input"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool3d_exception(self):
        input = torch.randn((2, 2, 3, 8, 8), dtype=torch.float).to("mlu")
        m = nn.MaxPool3d(kernel_size=(3, 3), stride=2)
        m = m.to("mlu")
        ref_msg = r"^max_pool3d: kernel_size must either be a single int,"
        ref_msg = ref_msg + r" or a tuple of three ints$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2, 2, 3, 8, 8), dtype=torch.float).to("mlu")
        m = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2))
        m = m.to("mlu")
        ref_msg = r"^max_pool3d: stride must either be omitted,"
        ref_msg = ref_msg + r" a single int, or a tuple of three ints$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2, 2, 3, 8, 8), dtype=torch.float).to("mlu")
        m = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), dilation=(1, 2, 2))
        m = m.to("mlu")
        ref_msg = r"^max_pool3d: dilation must be either a single int, or a tuple of three ints,"
        ref_msg = ref_msg + r" and cnnl pool3d only supports defalut dilation value$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((2, 2, 3, 80, 80), dtype=torch.float).to("mlu")
        m = nn.MaxPool3d(kernel_size=(100, 100, 16), stride=(2, 2, 2))
        m = m.to("mlu")
        ref_msg = r"^max_pool3d: The kernel size should be"
        ref_msg = ref_msg + r" smaller than 65535, while this kernel size is 160000"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

        input = torch.randn((0, 3, 4, 4, 4, 5), dtype=torch.float).to("mlu")
        m = torch.nn.MaxPool3d((2, 2, 2))
        ref_msg = r"non-empty 4D or 5D \(batch mode\) tensor expected for input"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            m(input)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_avgpool2d_bfloat16(self):
        shape_list = [(8, 16, 7, 7), (16, 6, 8, 16), (4, 23, 13, 64), (0, 2, 128, 128)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtypes = [
            torch.bfloat16,
        ]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtypes,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            include_pad,
            t,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=t).to(memory_format=memory_format)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            avg_pool = nn.AvgPool2d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            output_mlu = avg_pool(self.to_mlu(input_t))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # not dense
            output_cpu = avg_pool(input_cpu[:2, ...])
            output_mlu = avg_pool(self.to_mlu(input_t)[:2, ...])
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_maxpool2d_batch_bfloat16(self):
        in_shapes = [(4, 2, 12, 12), (0, 2, 128, 128)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [2, 3, 4]
        stride_v = [3, 4, 5]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtypes = [
            torch.bfloat16,
        ]

        loop_var = [
            memory_format_list,
            kernel_v,
            stride_v,
            padding_v,
            return_indices_v,
            ceil_mode_v,
            dtypes,
            in_shapes,
        ]
        for (
            memory_format,
            kernel,
            stride,
            padding,
            return_indices,
            ceil_mode,
            t,
            in_shape,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=t).to(memory_format=memory_format)
            input_cpu = input_t if (t != torch.half) else input_t.to(torch.float)
            output_cpu = F.max_pool2d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            output_mlu = F.max_pool2d(
                self.to_mlu(input_t),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )

            if return_indices == True:
                self.assertTensorsEqual(
                    output_cpu[0].float(),
                    output_mlu[0].cpu().float(),
                    3e-3,
                    use_MSE=True,
                )
                self.assertIs(output_mlu[1].dtype, torch.long)
            else:
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
                )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("37GB")
    def test_maxpool2d_batch_large(self):
        in_shape = (4, 1025, 1024, 1024)
        kernel = 32
        stride = 32
        padding = 0
        ceil_mode = False
        return_indices = True
        t = torch.float

        input_cpu = torch.randn(in_shape, dtype=t)
        output_cpu = F.max_pool2d(
            input_cpu,
            kernel,
            stride=stride,
            padding=padding,
            dilation=1,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

        output_mlu = F.max_pool2d(
            self.to_mlu(input_cpu),
            kernel,
            stride=stride,
            padding=padding,
            dilation=1,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

        if return_indices == True:
            self.assertTensorsEqual(
                output_cpu[0].float(), output_mlu[0].cpu().float(), 3e-3, use_MSE=True
            )
            self.assertIs(output_mlu[1].dtype, torch.long)
        else:
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("37GB")
    def test_maxpool3d_batch_large(self):
        in_shape = (4, 1025, 32, 32, 1024)
        kernel = 8
        stride = 4
        padding = 0
        ceil_mode = False
        return_indices = True
        t = torch.float

        input_cpu = torch.randn(in_shape, dtype=t)
        output_cpu = F.max_pool3d(
            input_cpu,
            kernel,
            stride=stride,
            padding=padding,
            dilation=1,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

        output_mlu = F.max_pool3d(
            self.to_mlu(input_cpu),
            kernel,
            stride=stride,
            padding=padding,
            dilation=1,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

        if return_indices == True:
            self.assertTensorsEqual(
                output_cpu[0].float(), output_mlu[0].cpu().float(), 3e-3, use_MSE=True
            )
            self.assertIs(output_mlu[1].dtype, torch.long)
        else:
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("21GB")
    def test_avgpool2d_batch_large(self):
        in_shape = (4, 1025, 1024, 1024)
        kernel = 5
        stride = 4
        padding = 0
        dtype = torch.half
        input_t = torch.randn(in_shape, dtype=dtype)
        input_cpu = input_t.to(torch.float)
        avg_pool = nn.AvgPool2d(kernel, stride=stride, padding=padding)
        output_cpu = avg_pool(input_cpu)
        output_mlu = avg_pool(self.to_mlu(input_t))
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("21GB")
    def test_avgpool3d_batch_large(self):
        in_shape = (4, 1025, 1024, 32, 32)
        kernel = 5
        stride = 4
        padding = 0
        dtype = torch.half
        input_t = torch.randn(in_shape, dtype=dtype)
        input_cpu = input_t.to(torch.float)
        avg_pool = nn.AvgPool3d(kernel, stride=stride, padding=padding)
        output_cpu = avg_pool(input_cpu)
        output_mlu = avg_pool(self.to_mlu(input_t))
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
