from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase, read_card_info  # pylint: disable=C0413

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestPoolbpOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool1d_backward_non_batch(self):
        shape_list = [(7, 7), (8, 16), (13, 64), (128, 128)]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4, None]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, dtype in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
            avg_pool = nn.AvgPool1d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)

            output_mlu = avg_pool(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)

            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 0.003, use_RAE=True
            )

            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool1d_backward_batch(self):
        shape_list = [(16, 7, 7), (16, 8, 16), (23, 13, 64), (0, 128, 128)]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4, None]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, dtype in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
            avg_pool = nn.AvgPool1d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)

            output_mlu = avg_pool(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)

            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 0.003, use_RAE=True
            )

            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool2d_backward_non_batch(self):
        shape_list = [(16, 7, 7), (6, 8, 16), (23, 13, 64), (4, 128, 128)]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4, None]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, dtype in product(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
            avg_pool = nn.AvgPool2d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)

            output_mlu = avg_pool(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)

            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 0.003, use_RAE=True
            )

            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool2d_backward_batch(self):
        shape_list = [(8, 16, 7, 7), (16, 6, 8, 16), (4, 23, 13, 64), (0, 2, 128, 128)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4, None]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            include_pad,
            dtype,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format)
            input_cpu = input_cpu if dtype != torch.half else input_cpu.to(torch.float)
            avg_pool = nn.AvgPool2d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad, retain_graph=True)

            output_mlu = avg_pool(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)

            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 0.003, use_RAE=True
            )

            # not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool1d_backward_non_batch(self):
        in_shapes = [(4, 4), (8, 8), (128, 128)]
        kernel_v = [3]
        stride_v = [2]
        padding_v = [1]
        ceil_mode_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]
        return_indices_v = [True, False]

        loop_var = [
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            in_shapes,
            dtype_list,
        ]
        for (
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            in_shape,
            dtype,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
            output_cpu = F.max_pool1d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_cpu, indices_cpu = output_cpu
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = F.max_pool1d(
                self.to_device(input_mlu),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_mlu, indices_mlu = output_mlu
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # test not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool2d_backward_non_batch(self):
        in_shapes = [(1, 8, 8), (2, 128, 128)]
        kernel_v = [3]
        stride_v = [2]
        padding_v = [1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            in_shapes,
            dtype_list,
        ]
        for (
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            in_shape,
            dtype,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
            output_cpu = F.max_pool2d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_cpu = output_cpu[0]
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = F.max_pool2d(
                self.to_device(input_mlu),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_mlu = output_mlu[0]
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # test not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool1d_backward_batch(self):
        in_shapes = [(1, 8, 8), (4, 16, 16), (0, 128, 128)]
        kernel_v = [3]
        stride_v = [2]
        padding_v = [1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            in_shapes,
            dtype_list,
        ]
        for (
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            in_shape,
            dtype,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
            output_cpu = F.max_pool1d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_cpu = output_cpu[0]
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = F.max_pool1d(
                self.to_device(input_mlu),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_mlu = output_mlu[0]
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # test not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool2d_backward_batch(self):
        in_shapes = [(1, 1, 8, 8), (2, 4, 16, 16), (0, 2, 128, 128)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3]
        stride_v = [2]
        padding_v = [1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            memory_format_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            in_shapes,
            dtype_list,
        ]
        for (
            memory_format,
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            in_shape,
            dtype,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format)
            input_cpu = input_cpu if dtype != torch.half else input_cpu.to(torch.float)
            output_cpu = F.max_pool2d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_cpu = output_cpu[0]
            grad = torch.ones(output_cpu.shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = F.max_pool2d(
                self.to_device(input_mlu),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_mlu = output_mlu[0]
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu_t.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # test not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool3d_backward_non_batch(self):
        shape_list = [(2048, 1, 7, 7), (192, 8, 28, 28)]
        kernel_v = [(1, 7, 7), (1, 3, 3)]
        stride_v = [(1, 1, 1), (1, 1, 1)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, include_pad, dtype in zip(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
            # test nn module
            avg_pool = nn.AvgPool3d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = avg_pool(self.to_mlu(input_mlu))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)
            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_avgpool3d_backward_batch(self):
        shape_list = [(12, 2048, 1, 7, 7), (12, 192, 8, 28, 28)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(1, 7, 7), (1, 3, 3)]
        stride_v = [(1, 1, 1), (1, 1, 1)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            include_pad,
            dtype,
        ) in zip(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format)
            input_cpu = input_cpu if dtype != torch.half else input_cpu.to(torch.float)
            # test nn module
            avg_pool = nn.AvgPool3d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = avg_pool(self.to_mlu(input_mlu))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu_t.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)
            # not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(
                o_shape[0], o_shape[1], o_shape[2], o_shape[3], o_shape[4] + 1
            ).to(memory_format=memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool3d_backward_non_batch(self):
        shape_list = [(2048, 2, 7, 7), (128, 8, 112, 112)]
        kernel_v = [(2, 1, 1), (2, 3, 3)]
        stride_v = [(1, 1, 1), (2, 2, 2)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            dtype_list,
        ]
        for in_shape, kernel, stride, padding, ceil_mode, return_indices, dtype in zip(
            *loop_var
        ):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_cpu = input_t if dtype != torch.half else input_t.to(torch.float)
            input_mlu = copy.deepcopy(input_t)
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
            if return_indices == True:
                output_cpu, _ = output_cpu
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = max_pool(self.to_mlu(input_mlu))
            if return_indices == True:
                output_mlu, _ = output_mlu
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # not dense
            input_t.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_maxpool3d_backward_batch(self):
        shape_list = [(12, 2048, 2, 7, 7), (12, 128, 8, 112, 112)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(2, 1, 1), (2, 3, 3)]
        stride_v = [(1, 1, 1), (2, 2, 2)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtype_list = [torch.float, torch.half, torch.double]

        loop_var = [
            shape_list,
            memory_format_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            dtype_list,
        ]
        for (
            in_shape,
            memory_format,
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            dtype,
        ) in zip(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format)
            input_cpu = input_cpu if dtype != torch.half else input_cpu.to(torch.float)
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
            if return_indices == True:
                output_cpu, _ = output_cpu
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = max_pool(self.to_mlu(input_mlu))
            if return_indices == True:
                output_mlu, _ = output_mlu
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu_t.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(
                o_shape[0], o_shape[1], o_shape[2], o_shape[3], o_shape[4] + 1
            ).to(memory_format=memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_avgpool2d_backward_bfloat16(self):
        shape_list = [(8, 16, 7, 7), (16, 6, 8, 16), (4, 23, 13, 64), (0, 2, 128, 128)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3, 4, 5]
        stride_v = [2, 3, 4, None]
        padding_v = [0, 1]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.bfloat16]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            include_pad,
            dtype,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format)
            avg_pool = nn.AvgPool2d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad, retain_graph=True)

            output_mlu = avg_pool(self.to_device(input_mlu))
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)

            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 0.003, use_RAE=True
            )

            # not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_maxpool2d_backward_bfloat16(self):
        in_shapes = [(1, 1, 8, 8), (2, 4, 16, 16)]
        memory_format_list = [torch.channels_last, torch.contiguous_format]
        kernel_v = [3,5]
        stride_v = [2,3]
        padding_v = [1]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtype_list = [torch.bfloat16]

        loop_var = [
            memory_format_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            in_shapes,
            dtype_list,
        ]
        for (
            memory_format,
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            in_shape,
            dtype,
        ) in product(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format)
            output_cpu = F.max_pool2d(
                input_cpu,
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_cpu = output_cpu[0]
            grad = torch.ones(output_cpu.shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = F.max_pool2d(
                self.to_device(input_mlu),
                kernel,
                stride=stride,
                padding=padding,
                dilation=1,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            )
            if return_indices == True:
                output_mlu = output_mlu[0]
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 3e-3, use_MSE=True)
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu_t.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # test not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_avgpool3d_backward_bfloat16(self):
        shape_list = [(12, 2048, 1, 7, 7), (12, 192, 8, 28, 28)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(1, 5, 5), (1, 3, 3)]
        stride_v = [(1, 1, 1), (1, 3, 3)]
        padding_v = [(0, 0, 0), (0, 1, 1)]
        ceil_mode_v = [False, True]
        include_pad_v = [False, True]
        dtype_list = [torch.bfloat16]

        loop_var = [
            memory_format_list,
            shape_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            include_pad_v,
            dtype_list,
        ]
        for (
            memory_format,
            in_shape,
            kernel,
            stride,
            padding,
            ceil_mode,
            include_pad,
            dtype,
        ) in zip(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format).to(torch.float)
            # test nn module
            avg_pool = nn.AvgPool3d(
                kernel,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=include_pad,
            )
            output_cpu = avg_pool(input_cpu)
            grad = torch.randn(output_cpu.shape, dtype=torch.float).to(
                memory_format=memory_format
            )
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = avg_pool(self.to_mlu(input_mlu))
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu_t.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)
            # not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(
                o_shape[0], o_shape[1], o_shape[2], o_shape[3], o_shape[4] + 1
            ).to(memory_format=memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_maxpool3d_backward_bfloat16(self):
        shape_list = [(12, 1024, 3, 7, 7), (12, 128, 8, 112, 112)]
        memory_format_list = [torch.channels_last_3d, torch.contiguous_format]
        kernel_v = [(2, 1, 1), (2, 3, 3)]
        stride_v = [(1, 1, 1), (1, 1, 1)]
        padding_v = [(0, 0, 0), (1, 1, 1)]
        ceil_mode_v = [False, True]
        return_indices_v = [False, True]
        dtype_list = [torch.bfloat16]

        loop_var = [
            shape_list,
            memory_format_list,
            kernel_v,
            stride_v,
            padding_v,
            ceil_mode_v,
            return_indices_v,
            dtype_list,
        ]
        for (
            in_shape,
            memory_format,
            kernel,
            stride,
            padding,
            ceil_mode,
            return_indices,
            dtype,
        ) in zip(*loop_var):
            input_t = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu_t = copy.deepcopy(input_t)
            input_mlu = input_mlu_t.to(memory_format=memory_format)
            input_cpu = input_t.to(memory_format=memory_format)
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
            if return_indices == True:
                output_cpu, _ = output_cpu
            grad = torch.ones(output_cpu.shape, dtype=torch.float)
            output_cpu.backward(grad, retain_graph=True)
            grad_cpu = input_t.grad

            output_mlu = max_pool(self.to_mlu(input_mlu))
            if return_indices == True:
                output_mlu, _ = output_mlu
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            output_mlu.backward(self.to_mlu_dtype(grad, dtype), retain_graph=True)
            grad_mlu = input_mlu_t.grad
            self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_RAE=True)

            # not dense
            input_t.grad.zero_()
            input_mlu_t.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(
                o_shape[0], o_shape[1], o_shape[2], o_shape[3], o_shape[4] + 1
            ).to(memory_format=memory_format)
            output_cpu.backward(grad[..., :-1])
            output_mlu.backward(self.to_mlu_dtype(grad, dtype)[..., :-1])
            self.assertTensorsEqual(
                input_t.grad.float(), input_mlu_t.grad.float(), 3e-3, use_RAE=True
            )

if __name__ == "__main__":
    unittest.main()
