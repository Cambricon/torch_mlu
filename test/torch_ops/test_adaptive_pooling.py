# pylint: disable=W0511
from __future__ import print_function

import unittest
import logging
import copy
from itertools import product
import sys
import os
import torch
from torch import nn


cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestAdaptivePoolingOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool1d(self):
        in_shape_list = [
            (0, 2),
            (1, 4),
            (16, 8),
            (0, 4, 15),
            (2, 3, 7),
            (8, 14, 14),
            (4, 23, 64),
            (64, 128, 128),
        ]
        out_shape_list = [(4), (10), (11), (2)]
        dtype_list = [torch.float, torch.double, torch.half]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        err = 3e-3
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool1d(out_shape)
            output_cpu = m(input_cpu.float())
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool1d_not_contiguous(self):
        in_shape_list = [(16, 14, 14), (4, 13, 64), (64, 128, 128)]
        out_shape_list = [(4), (9), (2)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., :-1]]
        list_list = [in_shape_list, out_shape_list, func_list, func_list]
        for in_shape, out_shape, in_func, out_func in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool1d(out_shape)
            output_cpu = m(in_func(input_cpu))
            output_mlu = m(in_func(self.to_device(input_mlu)))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad, retain_graph=True)
            output_mlu.backward(self.to_device(grad), retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test not dense grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2] + 1)
            output_cpu.backward(grad[..., :-1], retain_graph=True)
            output_mlu.backward(self.to_device(grad)[..., :-1], retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test channel last grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(self.convert_to_channel_last(grad))
            output_mlu.backward(self.convert_to_channel_last(self.to_device(grad)))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d(self):
        in_shape_list = [
            (0, 2, 3),
            (1, 4, 4),
            (16, 6, 8),
            (0, 4, 13, 15),
            (2, 3, 3, 7),
            (8, 16, 14, 14),
            (4, 23, 13, 64),
            (4, 64, 128, 128),
        ]
        out_shape_list = [(4, 4), (10, 7), (9, 11), (2, 2)]
        dtype_list = [torch.float, torch.double, torch.half]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        err = 3e-3
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool2d(out_shape)
            output_cpu = m(input_cpu.float())
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), err, use_MSE=True
            )
            # test out
            output_mlu = self.to_device(torch.randn(output_cpu.size(), dtype=dtype))
            with torch.no_grad():
                torch._C._nn.adaptive_avg_pool2d(
                    self.to_device(input_mlu), out_shape, out=output_mlu
                )
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d_not_contiguous(self):
        in_shape_list = [(8, 16, 14, 14), (4, 23, 13, 64), (4, 64, 128, 128)]
        out_shape_list = [(4, 4), (9, 11), (2, 2)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., :-1]]
        list_list = [in_shape_list, out_shape_list, func_list, func_list]
        for in_shape, out_shape, in_func, out_func in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool2d(out_shape)
            output_cpu = m(in_func(input_cpu))
            output_mlu = m(in_func(self.to_device(input_mlu)))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad, retain_graph=True)
            output_mlu.backward(self.to_device(grad), retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test not dense grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1)
            output_cpu.backward(grad[..., :-1], retain_graph=True)
            output_mlu.backward(self.to_device(grad)[..., :-1], retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test channel last grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(self.convert_to_channel_last(grad))
            output_mlu.backward(self.convert_to_channel_last(self.to_device(grad)))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test out
            output_mlu = out_func(self.to_device(torch.randn(output_cpu.size())))
            output_mlu_ptr = output_mlu.data_ptr()
            with torch.no_grad():
                torch._C._nn.adaptive_avg_pool2d(
                    in_func(self.to_device(input_mlu)), out_shape, out=output_mlu
                )
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
                self.assertEqual(output_mlu_ptr, output_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d_edge_case(self):
        err = 3e-3
        m = torch.nn.AdaptiveAvgPool2d((1, 5))
        input_cpu = torch.empty_strided(
            (1, 8, 1, 8), (8, 1, 64, 8), device="cpu"
        ).normal_()
        input_mlu = input_cpu.mlu()
        output_cpu = m(input_cpu)
        output_mlu = m(input_mlu)
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
        )

        m = torch.nn.AdaptiveAvgPool3d((5, 7, 4))
        input_cpu = torch.empty_strided(
            (1, 8, 8, 8, 8), (8, 1, 512, 64, 8), device="cpu"
        ).normal_()
        input_mlu = input_cpu.mlu()
        output_cpu = m(input_cpu)
        output_mlu = m(input_mlu)
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool3d(self):
        in_shape_list = [
            (0, 2, 3, 6),
            (1, 2, 3, 4),
            (6, 8, 16, 16),
            (16, 6, 8, 8),
            (0, 6, 4, 13, 15),
            (4, 8, 16, 14, 14),
            (4, 4, 23, 13, 64),
            (4, 4, 64, 128, 128),
        ]
        out_shape_list = [(4, 4, 4), (10, 7, 6), (9, 11, 10), (2, 2, 3)]
        dtype_list = [torch.float, torch.double, torch.half]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        err = 3e-3
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool3d(out_shape)
            output_cpu = m(input_cpu.float())
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), err, use_MSE=True
            )
            # test out
            output_mlu = self.to_device(torch.randn(output_cpu.size(), dtype=dtype))
            with torch.no_grad():
                torch._C._nn.adaptive_avg_pool3d(
                    self.to_device(input_mlu), out_shape, out=output_mlu
                )
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool3d_not_contiguous(self):
        in_shape_list = [(4, 8, 16, 14, 14), (4, 2, 23, 13, 64), (3, 4, 64, 128, 128)]
        out_shape_list = [(4, 4, 3), (9, 11, 10), (2, 2, 5)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., :-1]]
        list_list = [in_shape_list, out_shape_list, func_list, func_list]
        for in_shape, out_shape, in_func, out_func in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveAvgPool3d(out_shape)
            output_cpu = m(in_func(input_cpu))
            output_mlu = m(in_func(self.to_device(input_mlu)))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad, retain_graph=True)
            output_mlu.backward(self.to_device(grad), retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test not dense grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(
                o_shape[0], o_shape[1], o_shape[2], o_shape[3], o_shape[4] + 1
            )
            output_cpu.backward(grad[..., :-1], retain_graph=True)
            output_mlu.backward(self.to_device(grad)[..., :-1], retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test channel last grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(self.convert_to_channel_last(grad))
            output_mlu.backward(self.convert_to_channel_last(self.to_device(grad)))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test out
            output_mlu = out_func(self.to_device(torch.randn(output_cpu.size())))
            output_mlu_ptr = output_mlu.data_ptr()
            with torch.no_grad():
                torch._C._nn.adaptive_avg_pool3d(
                    in_func(self.to_device(input_mlu)), out_shape, out=output_mlu
                )
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
                self.assertEqual(output_mlu_ptr, output_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool1d(self):
        in_shape_list = [(8, 14, 14), (16, 8), (4, 13, 64), (8, 16)]
        out_shape_list = [(4), (10), (11)]
        dtype_list = [torch.float, torch.half, torch.double]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveMaxPool1d(out_shape)
            output_cpu = m(input_cpu.float())
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )
        # test indices
        input_cpu = torch.arange(16).view(1, 4, 4).float()
        m = nn.AdaptiveMaxPool1d((2), return_indices=True)
        # Different with the origin CPU/GPU ops, the max indices returned by
        # MLU adaptive_max_pool2d_with_indices are local max indices inside the kernel
        output_cpu, _ = m(input_cpu)
        output_mlu, indices_mlu = m(self.to_device(input_cpu))
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
        self.assertTensorsEqual(
            torch.tensor([[[1, 1], [1, 1], [1, 1], [1, 1]]]), indices_mlu.cpu(), 0
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool1d_not_contiguous(self):
        in_shape_list = [(16, 14, 14), (4, 13, 64)]
        out_shape_list = [(4), (9)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., :-1]]
        list_list = [in_shape_list, out_shape_list, func_list, func_list]
        for in_shape, out_shape, in_func, out_func in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveMaxPool1d(out_shape)
            output_cpu = m(in_func(input_cpu.float()))
            output_mlu = m(in_func(self.to_device(input_mlu)))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad, retain_graph=True)
            output_mlu.backward(self.to_device(grad), retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test not dense grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2] + 1)
            output_cpu.backward(grad[..., :-1], retain_graph=True)
            output_mlu.backward(self.to_device(grad)[..., :-1], retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test channel last grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(self.convert_to_channel_last(grad))
            output_mlu.backward(self.convert_to_channel_last(self.to_device(grad)))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d(self):
        in_shape_list = [(8, 16, 14, 14), (16, 6, 8), (4, 23, 13, 64), (6, 8, 16)]
        out_shape_list = [(4, 4), (10, 7), (9, 11)]
        dtype_list = [torch.float, torch.half, torch.double]
        list_list = [in_shape_list, out_shape_list, dtype_list]
        for in_shape, out_shape, dtype in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, dtype=dtype, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveMaxPool2d(out_shape)
            output_cpu = m(input_cpu.float())
            output_mlu = m(self.to_device(input_mlu))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad)
            output_mlu.backward(self.to_device(grad))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )
            # test out
            output_mlu = self.to_device(torch.randn(output_cpu.size(), dtype=dtype))
            index_mlu = self.to_device(torch.randint(-10, 10, output_cpu.size()))
            with torch.no_grad():
                torch._C._nn.adaptive_max_pool2d(
                    self.to_device(input_mlu), out_shape, out=[output_mlu, index_mlu]
                )
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
        # test indices
        input_cpu = torch.arange(16).view(1, 4, 4).float()
        m = nn.AdaptiveMaxPool2d((2, 2), return_indices=True)
        # Different with the origin CPU/GPU ops, the max indices returned by
        # MLU adaptive_max_pool2d_with_indices are local max indices inside the kernel
        output_cpu, _ = m(input_cpu)
        output_mlu, indices_mlu = m(self.to_device(input_cpu))
        self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0)
        self.assertTensorsEqual(torch.tensor([[[3, 3], [3, 3]]]), indices_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d_not_contiguous(self):
        in_shape_list = [(8, 16, 14, 14), (4, 23, 13, 64)]
        out_shape_list = [(4, 4), (9, 11)]
        func_list = [lambda x: x, self.convert_to_channel_last, lambda x: x[..., :-1]]
        list_list = [in_shape_list, out_shape_list, func_list, func_list]
        for in_shape, out_shape, in_func, out_func in product(*list_list):
            # test forward
            input_cpu = torch.randn(in_shape, requires_grad=True)
            input_mlu = copy.deepcopy(input_cpu)
            m = nn.AdaptiveMaxPool2d(out_shape)
            output_cpu = m(in_func(input_cpu.float()))
            output_mlu = m(in_func(self.to_device(input_mlu)))
            self.assertTensorsEqual(
                output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
            )
            # test backward
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(grad, retain_graph=True)
            output_mlu.backward(self.to_device(grad), retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test not dense grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            o_shape = output_cpu.size()
            grad = torch.randn(o_shape[0], o_shape[1], o_shape[2], o_shape[3] + 1)
            output_cpu.backward(grad[..., :-1], retain_graph=True)
            output_mlu.backward(self.to_device(grad)[..., :-1], retain_graph=True)
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test channel last grad backward
            input_cpu.grad.zero_()
            input_mlu.grad.zero_()
            grad = torch.randn(output_cpu.size())
            output_cpu.backward(self.convert_to_channel_last(grad))
            output_mlu.backward(self.convert_to_channel_last(self.to_device(grad)))
            self.assertTensorsEqual(
                input_cpu.grad.float(), input_mlu.grad.float(), 3e-3, use_MSE=True
            )

            # test out
            output_mlu = out_func(self.to_device(torch.randn(output_cpu.size())))
            index_mlu = out_func(
                self.to_device(torch.randint(-10, 10, output_cpu.size()))
            )
            output_mlu_ptr = output_mlu.data_ptr()
            index_mlu_ptr = index_mlu.data_ptr()
            with torch.no_grad():
                torch._C._nn.adaptive_max_pool2d(
                    in_func(self.to_device(input_mlu)),
                    out_shape,
                    out=[output_mlu, index_mlu],
                )
                self.assertTensorsEqual(
                    output_cpu.float(), output_mlu.cpu().float(), 3e-3, use_MSE=True
                )
                self.assertEqual(output_mlu_ptr, output_mlu.data_ptr())
                self.assertEqual(index_mlu_ptr, index_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d_zero_output_size(self):
        m = torch.nn.AdaptiveMaxPool2d((1, 0))
        x = torch.randn(1, 2, 3, 4, dtype=torch.float)
        out_cpu = m(x)
        out_mlu = m.to("mlu")(x.to("mlu"))
        self.assertTensorsEqual(out_cpu, out_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool3d_backward_zero_intput(self):
        grad_output = torch.randn(0, 5, 5, 5)
        input_t = torch.randn(0, 10, 10, 10)
        cpu_out = torch.ops.aten._adaptive_avg_pool3d_backward(grad_output, input_t)
        mlu_out = torch.ops.aten._adaptive_avg_pool3d_backward(
            grad_output.to("mlu"), input_t.to("mlu")
        )
        assert cpu_out.size() == mlu_out.size()

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool2d_exception(self):
        input = torch.randn((2, 3), dtype=torch.float).to("mlu")
        m = nn.AdaptiveAvgPool2d(7)
        ref_msg = (
            r"^adaptive_avg_pool2d\(\): Expected 3D or 4D tensor," " but got: \[2, 3\]$"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

        input = torch.randn((2, 0, 4), dtype=torch.float).to("mlu")
        m = nn.AdaptiveAvgPool2d(7)
        ref_msg = r"^adaptive_avg_pool2d\(\): Expected input to have"
        ref_msg = ref_msg + " non-zero size for non-batch dimensions, but input"
        ref_msg = ref_msg + " has sizes \[2, 0, 4\] with dimension 1 being empty$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

        inputs = torch.randn(
            (32, 16, 120, 120), dtype=torch.float, requires_grad=True
        ).to("mlu")
        m = torch.nn.AdaptiveAvgPool2d((2, 2, 2))
        ref_msg = r"^adaptive_avg_pool2d: output_size must be 2$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(inputs)

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_avg_pool3d_exception(self):
        input = torch.randn((2, 3, 4), dtype=torch.float).to("mlu")
        m = nn.AdaptiveAvgPool3d(7)
        ref_msg = (
            r"^cnnl_adaptive_avg_pool3d\(\): Expected 4D or 5D tensor,"
            " but got \[2, 3, 4\]$"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

        input = torch.randn((2, 0, 4, 4), dtype=torch.float).to("mlu")
        m = nn.AdaptiveAvgPool3d(7)
        ref_msg = (
            r"^cnnl_adaptive_avg_pool3d\(\): Expected input to have"
            " non-zero size for non-batch dimensions, but input"
            " has sizes \[2, 0, 4, 4\] with dimension 1 being empty$"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

        inputs = torch.randn(
            (32, 16, 120, 120), dtype=torch.float, requires_grad=True
        ).to("mlu")
        m = torch.nn.AdaptiveAvgPool3d((2, 2))
        ref_msg = r"^adaptive_avg_pool3d: output_size must be 3$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(inputs)

        inputs = torch.randn(
            (32, 16, 120, 120), dtype=torch.float, requires_grad=True
        ).to("mlu")
        m = torch.nn.AdaptiveAvgPool3d((2, 2, 2))
        ref_msg = (
            "^The internal kernel size for cnnl_adaptive_avg_pool3d_backward_out"
            " should be smaller than 3582, while this kernel size is 3844$"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(inputs)
            grad = torch.randn(output.size()).to("mlu")
            output.backward(grad)

    # @unittest.skip("not test")
    @testinfo()
    def test_adaptive_max_pool2d_exception(self):
        input = torch.randn((2, 3), dtype=torch.float).to("mlu")
        m = nn.AdaptiveMaxPool2d(7)
        ref_msg = (
            r"adaptive_max_pool2d\(\): Expected 3D or 4D tensor, but got: \[2, 3\]"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)  # pylint: disable=W0612

        input = torch.randn((2, 0, 3, 7), dtype=torch.float).to("mlu")
        m = nn.AdaptiveMaxPool2d(7)
        ref_msg = r"^adaptive_max_pool2d\(\): Expected input to have non-zero size "
        ref_msg = ref_msg + r"for non-batch dimensions, but input has sizes "
        ref_msg = ref_msg + r"\[2, 0, 3, 7\] with dimension 1 being empty$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

        input = torch.randn((2, 2, 3, 7), dtype=torch.float).to("mlu")
        m = nn.AdaptiveMaxPool2d((2, 2, 2))
        ref_msg = (
            r"^adaptive_max_pool2d\(\): internal error:"
            " output_size\.size\(\) must be 2$"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(input)

        inputs = torch.randn((32, 16, 120, 120), dtype=torch.float).to("mlu")
        m = torch.nn.AdaptiveMaxPool2d((2, 2))
        ref_msg = (
            "^The internal kernel size for adaptive_max_pool2d_out_mlu should be smaller than 3582, "
            + "while this kernel size is 3844$"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            output = m(inputs)  # pylint: disable=W0612

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_adaptive_avg_pooling_bfloat16(self):
        torch.manual_seed(17)

        input_shape = [2, 3, 7]
        output_shape = [4]
        input_cpu = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
        input_mlu = copy.deepcopy(input_cpu)
        m = nn.AdaptiveAvgPool1d(output_shape)
        output_cpu = m(input_cpu)
        output_mlu = m(self.to_device(input_mlu))
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
        )
        # test backward
        grad = torch.randn(output_cpu.size())
        output_cpu.backward(grad)
        output_mlu.backward(grad.to(torch.bfloat16).to("mlu"))
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.float(), 0.003, use_MSE=True
        )

        input_shape = [2, 3, 3, 7]
        output_shape = [4, 4]
        input_cpu = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
        input_mlu = copy.deepcopy(input_cpu)
        m = nn.AdaptiveAvgPool2d(output_shape)
        output_cpu = m(input_cpu)
        output_mlu = m(self.to_device(input_mlu))
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
        )
        # test backward
        grad = torch.randn(output_cpu.size())
        output_cpu.backward(grad)
        output_mlu.backward(grad.to(torch.bfloat16).to("mlu"))
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.float(), 0.003, use_MSE=True
        )

        input_shape = [2, 3, 3, 4, 7]
        output_shape = [4, 4, 5]
        input_cpu = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
        input_mlu = copy.deepcopy(input_cpu)
        m = nn.AdaptiveAvgPool3d(output_shape)
        # adaptive_avg_pool3d_cpu not implemented for BFloat16
        output_cpu = m(input_cpu.float())
        output_mlu = m(self.to_device(input_mlu))
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
        )
        # test backward
        grad = torch.randn(output_cpu.size(), dtype=torch.bfloat16)
        output_cpu.backward(grad.float())
        output_mlu.backward(grad.to("mlu"))
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.float(), 0.003, use_MSE=True
        )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_adaptive_max_pooling_bfloat16(self):
        input_shape = [2, 3, 7]
        output_shape = [4]
        input_cpu = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
        input_mlu = copy.deepcopy(input_cpu)
        m = nn.AdaptiveMaxPool1d(output_shape)
        output_cpu = m(input_cpu)
        output_mlu = m(self.to_device(input_mlu))
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
        )
        # test backward
        grad = torch.randn(output_cpu.size())
        output_cpu.backward(grad)
        output_mlu.backward(grad.to(torch.bfloat16).to("mlu"))
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.float(), 0.003, use_MSE=True
        )

        input_shape = [2, 3, 3, 7]
        output_shape = [4, 4]
        input_cpu = torch.randn(input_shape, dtype=torch.bfloat16, requires_grad=True)
        input_mlu = copy.deepcopy(input_cpu)
        m = nn.AdaptiveMaxPool2d(output_shape)
        output_cpu = m(input_cpu)
        output_mlu = m(self.to_device(input_mlu))
        self.assertTensorsEqual(
            output_cpu.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
        )
        # test backward
        grad = torch.randn(output_cpu.size())
        output_cpu.backward(grad)
        output_mlu.backward(grad.to(torch.bfloat16).to("mlu"))
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
