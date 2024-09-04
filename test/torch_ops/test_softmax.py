from __future__ import print_function
from itertools import product

import sys
import os
import copy
import unittest
import logging
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
)

logging.basicConfig(level=logging.DEBUG)


class LogSoftMaxFunc(torch.autograd.Function):  # pylint: disable=W0223
    @staticmethod
    def forward(ctx, x, dim, result=None):
        if x.device.type == "mlu":
            result = torch.log_softmax(x, dim)
        else:
            if result is None:
                logging.error("logsoftmaxbackward requires result!!")
        ctx.save_for_backward(x, result)
        ctx.dim = dim
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, result = ctx.saved_tensors
        dim = ctx.dim
        grad = torch._log_softmax_backward_data(grad_output, result, dim, x.dtype)
        return grad


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test__softmax_half_to_float(self):
        _softmax = torch._softmax
        shapes = [
            (2, 3, 5),
            (7, 8, 9, 10),
            (2, 0, 3, 5),
            (10,),
            (10, 15),
            (10, 20, 30, 40, 50),
        ]
        for shape, option in product(*[shapes, [True, False]]):
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = _softmax(x, dim, half_to_float=False)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x_mlu = self.to_mlu_dtype(x, torch.half if option else torch.float)
                out_mlu = _softmax(x_mlu, dim, half_to_float=option)
                x.grad.zero_()
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test__softmax_half_to_float_out(self):
        _softmax = torch._softmax
        shapes = [
            [(2, 3, 5), (2, 3, 5)],
            [(7, 8, 9, 10), (9, 4)],
            [(10,), (3, 5, 7)],
            [(2, 0, 3, 5), (9, 4)],
            [(10, 15), (10, 15)],
        ]
        for [shape, out_shape], option in product(*[shapes, [True, False]]):
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.randn(out_shape, dtype=torch.float)
                out_mlu = self.to_mlu(out_cpu)
                res_cpu = _softmax(x, dim, half_to_float=False, out=out_cpu)

                x_mlu = self.to_mlu_dtype(x, torch.half if option else torch.float)

                res_mlu = _softmax(x_mlu, dim, half_to_float=option, out=out_mlu)

                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax_half_to_float(self):
        _softmax = torch.softmax
        shapes = [
            [(2, 3, 5), (2, 3, 5)],
            [(7, 8, 9, 10), (9, 4)],
            [(10,), (3, 5, 7)],
            [(2, 0, 3, 5), (9, 4)],
            [(10, 15), (10, 15)],
        ]
        dtypes = [(torch.half, torch.float), (torch.float, torch.float)]
        for [shape, out_shape], [dtype1, dtype2] in product(*[shapes, dtypes]):
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=dtype1)
                res_cpu = _softmax(x, dim, dtype=dtype2)
                res_mlu = _softmax(x.mlu(), dim, dtype=dtype2)

                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax_half_to_float_out(self):
        _softmax = torch.softmax
        shapes = [
            [(2, 3, 5), (2, 3, 5)],
            [(7, 8, 9, 10), (9, 4)],
            [(2, 3, 4, 5), (2, 3, 4, 5)],
            [(10,), (3, 5, 7)],
            [(2, 0, 3, 5), (9, 4)],
            [(10, 15), (10, 15)],
        ]
        dtypes = [(torch.half, torch.float), (torch.float, torch.float)]
        channel_last_format = [True, False]
        for [shape, out_shape], [dtype1, dtype2], last_memory_format in product(
            *[shapes, dtypes, channel_last_format]
        ):
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=dtype1)
                out_cpu = (
                    torch.randn(out_shape, dtype=dtype2).to(
                        memory_format=torch.channels_last
                    )
                    if last_memory_format and len(out_shape) == 4
                    else torch.randn(out_shape, dtype=dtype2)
                )
                out_mlu = self.to_mlu(out_cpu)
                res_cpu = _softmax(x, dim, dtype=dtype2, out=out_cpu)
                res_mlu = _softmax(x.mlu(), dim, dtype=dtype2, out=out_mlu)

                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test__softmax_half_to_float_0dim(self):
        _softmax = torch._softmax
        for option in [True, False]:
            x = torch.tensor(999, dtype=torch.float, requires_grad=True)
            dim = 0
            out_cpu = _softmax(x, dim, half_to_float=False)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)

            x_mlu = self.to_mlu_dtype(x, torch.half if option else torch.float)
            out_mlu = _softmax(x_mlu, dim, half_to_float=option)
            x.grad.zero_()
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
            out_mlu.backward(self.to_mlu(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(
                grad_cpu, grad_mlu.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test__softmax_half_to_float_out_0dim(self):
        _softmax = torch._softmax
        for option in [True, False]:
            x = torch.tensor(999, dtype=torch.float)
            out_cpu = torch.tensor((64, 32), dtype=torch.float)
            out_mlu = self.to_mlu(out_cpu)
            dim = 0
            res_cpu = _softmax(x, dim, half_to_float=False, out=out_cpu)
            x_mlu = self.to_mlu_dtype(x, torch.half if option else torch.float)
            res_mlu = _softmax(x_mlu, dim, half_to_float=option, out=out_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax(self):
        shapes = [
            (2, 3, 4, 5, 7, 8, 9, 11),
            (2, 3, 4, 5),
            (2, 0, 3, 5),
            (2, 3, 4),
            (2, 3),
            (2,),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                for dim in range(len(shape)):
                    y_mlu = torch.nn.functional.softmax(x_mlu, dim)
                    y_cpu = torch.nn.functional.softmax(x_cpu, dim)
                    self.assertTensorsEqual(
                        y_cpu, y_mlu.cpu().float(), err, use_MSE=True
                    )

        for data_type, err in dtype_list:
            x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
            x_mlu = self.to_mlu_dtype(x_cpu, data_type)
            dims = [-3, -2, -1, 0, 1, 2, 3]
            for i in range(len(dims)):  # pylint: disable=C0200
                y_mlu = torch.nn.functional.softmax(x_mlu, dims[i])
                y_cpu = torch.nn.functional.softmax(x_cpu, dims[i])
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax_channels_last(self):
        shapes = [(2, 3, 4, 5), (2, 3, 24, 30), (2, 0, 3, 5), (1, 1, 1, 30)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float).to(
                    memory_format=torch.channels_last
                )
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                for dim in range(len(shape)):
                    y_mlu = torch.nn.functional.softmax(x_mlu, dim)
                    y_cpu = torch.nn.functional.softmax(x_cpu, dim)
                    self.assertTensorsEqual(
                        y_cpu, y_mlu.cpu().float(), err, use_MSE=True
                    )

        for data_type, err in dtype_list:
            x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
            x_mlu = self.to_mlu_dtype(x_cpu, data_type)
            dims = [-3, -2, -1, 0, 1, 2, 3]
            for i in range(len(dims)):  # pylint: disable=C0200
                y_mlu = torch.nn.functional.softmax(x_mlu, dims[i])
                y_cpu = torch.nn.functional.softmax(x_cpu, dims[i])
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax_not_dense(self):
        shapes = [(2, 3, 4, 5), (2, 3, 4), (2, 0, 3, 5), (2, 3)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                for dim in range(len(shape)):
                    y_mlu = torch.nn.functional.softmax(x_mlu[:, :2], dim)
                    y_cpu = torch.nn.functional.softmax(x_cpu[:, :2], dim)
                    self.assertTensorsEqual(
                        y_cpu, y_mlu.cpu().float(), err, use_MSE=True
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax_backward(self):
        shapes = [(2, 3, 4, 5), (2, 0, 3, 5), (2, 3, 4), (2, 3), (2,), ()]
        for shape in shapes:
            for dim in range(max(len(shape), 1)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = x.softmax(dim)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()
                out_mlu = self.to_mlu(x).softmax(dim)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_MSE=True)
        # test empty tensor
        x = torch.randn([], dtype=torch.float, requires_grad=True)
        out_cpu = x.softmax(0)
        grad = torch.randn(out_cpu.shape, dtype=torch.float)
        out_cpu.backward(grad)
        grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu = self.to_mlu(x).softmax(0)
        out_mlu.backward(self.to_mlu(grad))
        grad_mlu = copy.deepcopy(x.grad)
        self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nn_Softmax(self):
        shapes = [
            (2, 3, 4, 5, 7, 8, 9, 11),
            (2, 3, 4, 5),
            (2, 0, 3, 5),
            (2, 3, 4),
            (2, 3),
            (2,),
        ]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                for dim in range(len(shape)):
                    model = torch.nn.Softmax(dim=dim)
                    model_mlu = torch.nn.Softmax(dim=dim).to("mlu")
                    y_cpu = model(x_cpu)
                    y_mlu = model_mlu(x_mlu)
                    self.assertTensorsEqual(
                        y_cpu, y_mlu.cpu().float(), err, use_MSE=True
                    )

        for data_type, err in dtype_list:
            x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
            x_mlu = self.to_mlu_dtype(x_cpu, data_type)
            dims = [-3, -2, -1, 0, 1, 2, 3]
            for i in range(len(dims)):  # pylint: disable=C0200
                y_mlu = torch.nn.Softmax(dims[i])(x_mlu)
                y_cpu = torch.nn.Softmax(dims[i])(x_cpu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_nn_Softmax2d(self):
        shapes = [(2, 3, 4, 5), (2, 0, 3, 5), (2, 3, 4)]
        dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float, requires_grad=True)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                model = torch.nn.Softmax2d()
                model_mlu = torch.nn.Softmax().to("mlu")
                y_cpu = model(x_cpu)
                y_mlu = model_mlu(x_mlu)
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_logsoftmax(self):
        shapes = [(64, 1000), (16, 5, 7), (2, 3, 4, 5), (2, 0, 3, 5), (2,), ()]
        log_softmax_cpu = LogSoftMaxFunc()
        log_softmax_mlu = LogSoftMaxFunc()
        for shape in shapes:
            for dim in range(max(len(shape), 1)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_mlu = log_softmax_mlu.apply(self.to_mlu(x), dim)
                out_cpu = log_softmax_cpu.apply(x, dim, out_mlu.cpu())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                grad_cpu = out_cpu.grad_fn.apply(grad)
                grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
                self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)
        dims = [-3, -2, -1, 0, 1, 2, 3]
        for i in range(len(dims)):  # pylint: disable=C0200,W0612
            x = torch.randn(2, 3, 4, 5, dtype=torch.float, requires_grad=True)
            out_mlu = log_softmax_mlu.apply(self.to_mlu(x), dim)
            out_cpu = log_softmax_cpu.apply(x, dim, out_mlu.cpu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_cpu = out_cpu.grad_fn.apply(grad)
            grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
            self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_log_softmax_half_to_float(self):
        _log_softmax = torch.log_softmax
        shapes = [
            [(2, 3, 5), (2, 3, 5)],
            [(7, 8, 9, 10), (9, 4)],
            [(10,), (3, 5, 7)],
            [(2, 0, 3, 5), (9, 4)],
            [(10, 15), (10, 15)],
        ]
        dtypes = [(torch.half, torch.float), (torch.float, torch.float)]
        for [shape, out_shape], [dtype1, dtype2] in product(*[shapes, dtypes]):
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=dtype1)
                res_cpu = _log_softmax(x, dim, dtype=dtype2)
                res_mlu = _log_softmax(x.mlu(), dim, dtype=dtype2)

                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_log_softmax_half_to_float_out(self):
        _log_softmax = torch.log_softmax
        shapes = [
            [(2, 3, 5), (2, 3, 5)],
            [(7, 8, 9, 10), (9, 4)],
            [(2, 3, 4, 5), (2, 3, 4, 5)],
            [(10,), (3, 5, 7)],
            [(2, 0, 3, 5), (9, 4)],
            [(10, 15), (10, 15)],
        ]
        dtypes = [(torch.half, torch.float), (torch.float, torch.float)]
        channel_last_format = [True, False]
        for [shape, out_shape], [dtype1, dtype2], last_memory_format in product(
            *[shapes, dtypes, channel_last_format]
        ):
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=dtype1)
                out_cpu = (
                    torch.randn(out_shape, dtype=dtype2).to(
                        memory_format=torch.channels_last
                    )
                    if last_memory_format and len(out_shape) == 4
                    else torch.randn(out_shape, dtype=dtype2)
                )
                out_mlu = self.to_mlu(out_cpu)
                res_cpu = _log_softmax(x, dim, dtype=dtype2, out=out_cpu)
                res_mlu = _log_softmax(x.mlu(), dim, dtype=dtype2, out=out_mlu)

                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_channels_last(self):
        shapes = [(2, 3, 4, 5), (2, 3, 24, 30), (2, 0, 3, 5), (1, 1, 1, 30)]
        log_softmax_cpu = LogSoftMaxFunc()
        log_softmax_mlu = LogSoftMaxFunc()
        for shape in shapes:
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True).to(
                    memory_format=torch.channels_last
                )
                out_mlu = log_softmax_mlu.apply(self.to_mlu(x), dim)
                out_cpu = log_softmax_cpu.apply(x, dim, out_mlu.cpu())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                grad_cpu = out_cpu.grad_fn.apply(grad)
                grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
                self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_not_dense(self):
        shapes = [(64, 1000), (16, 5, 7), (2, 3, 4, 5), (2, 0, 3, 5)]
        log_softmax_cpu = LogSoftMaxFunc()
        log_softmax_mlu = LogSoftMaxFunc()
        for shape in shapes:
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_mlu = log_softmax_mlu.apply(self.to_mlu(x)[:, :2], dim)
                out_cpu = log_softmax_cpu.apply(x[:, :2], dim, out_mlu.cpu())
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                grad_cpu = out_cpu.grad_fn.apply(grad)
                grad_mlu = out_mlu.grad_fn.apply(self.to_mlu(grad))
                self.assertTensorsEqual(grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test__log_softmax_half_to_float_out(self):
        _softmax = torch._log_softmax
        shapes = [
            [(2, 3, 5), (2, 3, 5)],
            [(7, 8, 9, 10), (9, 4)],
            [(10,), (3, 5, 7)],
            [(2, 0, 5, 3), (9, 4)],
            [(10, 15), (10, 15)],
        ]
        for [shape, out_shape], option in product(*[shapes, [True, False]]):
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float)
                out_cpu = torch.randn(out_shape, dtype=torch.float)
                out_mlu = self.to_mlu(out_cpu)
                res_cpu = _softmax(x, dim, half_to_float=False, out=out_cpu)
                x_mlu = self.to_mlu_dtype(x, torch.half if option else torch.float)
                res_mlu = _softmax(x_mlu, dim, half_to_float=option, out=out_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    res_cpu, res_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_special_softmax(self):
        _softmax = torch.special.softmax
        shapes = [
            (2, 3, 5),
            (7, 8, 9, 10),
            (2, 0, 3, 5),
            (10,),
            (10, 15),
            (10, 20, 30, 40, 50),
        ]
        for shape in shapes:
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = _softmax(x, dim)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x_mlu = self.to_mlu_dtype(x, torch.float)
                out_mlu = _softmax(x_mlu, dim)
                x.grad.zero_()
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_special_logsoftmax(self):
        _softmax = torch.special.log_softmax
        shapes = [
            (2, 3, 5),
            (7, 8, 9, 10),
            (2, 0, 3, 5),
            (10,),
            (10, 15),
            (10, 20, 30, 40, 50),
        ]
        for shape in shapes:
            for dim in range(len(shape)):
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = _softmax(x, dim)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x_mlu = self.to_mlu_dtype(x, torch.float)
                out_mlu = _softmax(x_mlu, dim)
                x.grad.zero_()
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_and_softmax_backward_non_dense(self):
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

        for func in [
            torch.ops.aten._log_softmax_backward_data,
            torch.ops.aten._softmax_backward_data,
        ]:
            grad_output = torch.randn(16, 32, dtype=torch.float32)
            output = torch.randn(16, 32, dtype=torch.float32)
            cpu_out = func(grad_output, output, 1, torch.float32)
            device_out = func(
                to_non_dense(grad_output.mlu(), grad_output.dim() // 2),
                to_non_dense(output.mlu(), output.dim() // 2),
                1,
                torch.float32,
            )
            self.assertTensorsEqual(cpu_out, device_out.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_logsoftmax_exception(self):
        a = torch.randn(3, dtype=torch.float).to("mlu")
        ref_msg = r"^conversion is supported for Half type only$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch._log_softmax(a, dim=0, half_to_float=True)

        a_mlu = torch.randn(2, 10, dtype=torch.float).to("mlu")
        result = torch.log_softmax(a_mlu, dim=1)
        grad = torch.randn(2, 10, dtype=torch.half).to("mlu")
        ref_msg = r"^expected input and grad types to match,"
        ref_msg += " or input to be at::Half and grad to be at::Float$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch._log_softmax_backward_data(grad, result, 1, torch.float)

    # @unittest.skip("not test")
    @testinfo()
    def test_softmax_exception(self):
        _softmax = torch._softmax
        a = torch.randn(3, dtype=torch.float).to("mlu")
        ref_msg = r"^conversion is supported for Half type only$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            _softmax(a, dim=0, half_to_float=True)

        a_mlu = torch.randn(2, 10, dtype=torch.float).to("mlu")
        result = torch._softmax(a_mlu, dim=1, half_to_float=False)
        grad = torch.randn(2, 10, dtype=torch.half).to("mlu")
        ref_msg = r"^expected input and grad types to match,"
        ref_msg += " or input to be at::Half and grad to be at::Float$"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch._softmax_backward_data(grad, result, 1, torch.float)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("48GB")
    def test_softmax_large(self):
        shapes = [(48, 4096, 13725), (1, 4096 * 48 * 13725)]
        dtype_list = [(torch.half, 3e-3)]
        dim = 1
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = x.softmax(dim)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()
                x_mlu = self.to_mlu_dtype(x, data_type)
                out_mlu = x_mlu.softmax(dim)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu().float(), err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_softmax_bfloat16(self):
        softmax_list = [
            torch._softmax,
            torch.special.softmax,
            torch._log_softmax,
            torch.special.log_softmax,
        ]
        for softmax_ in softmax_list:
            x = torch.randn((7, 8, 9, 10), dtype=torch.bfloat16, requires_grad=True)
            if softmax_ in [torch.special.softmax, torch.special.log_softmax]:
                out_cpu = softmax_(x, -1)
            else:
                # CPU only support False
                out_cpu = softmax_(x, -1, half_to_float=False)
            grad = torch.randn(out_cpu.shape, dtype=torch.bfloat16)
            out_cpu.backward(grad)
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            x_mlu = x.mlu()
            if softmax_ in [torch.special.softmax, torch.special.log_softmax]:
                out_mlu = softmax_(x_mlu, -1)
            else:
                out_mlu = softmax_(x_mlu, -1, half_to_float=False)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
            out_mlu.backward(self.to_mlu(grad))
            grad_mlu = copy.deepcopy(x.grad)
            self.assertTensorsEqual(
                grad_cpu, grad_mlu.cpu().float(), 0.003, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("48GB")
    def test_softmax_large_bfloat16(self):
        shapes = [(48, 4096, 13725)]
        dtype_list = [(torch.bfloat16, 3e-3)]
        dim = 1
        for shape in shapes:
            for data_type, err in dtype_list:
                x_cpu = torch.randn(shape, dtype=torch.float)
                x_mlu = self.to_mlu_dtype(x_cpu, data_type)
                x = torch.randn(shape, dtype=torch.float, requires_grad=True)
                out_cpu = x.log_softmax(dim)
                grad = torch.randn(out_cpu.shape, dtype=torch.float)
                out_cpu.backward(grad)
                grad_cpu = copy.deepcopy(x.grad)
                x.grad.zero_()
                x_mlu = self.to_mlu_dtype(x, data_type)
                out_mlu = x_mlu.log_softmax(dim)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), err, use_MSE=True
                )
                out_mlu.backward(self.to_mlu(grad))
                grad_mlu = copy.deepcopy(x.grad)
                self.assertTensorsEqual(
                    grad_cpu, grad_mlu.cpu().float(), err, use_MSE=True
                )


if __name__ == "__main__":
    run_tests()
