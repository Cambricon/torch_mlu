from __future__ import print_function
import logging
import sys
import os
from itertools import product
import unittest
import copy
import torch
from torch import nn
import torch.autograd

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413,C0411


def to_mlu(tensor_cpu):
    return tensor_cpu.mlu()


bias_lst = [True, False]
N_lst = [8]
Ci_lst = [8]
Co_lst = [4]
D_lst = [18]
DHW_lst = [20]
padding_lst = [0, 1]
kernel_lst = [(1, 2, 2)]
stride_lst = [(1, 2, 2)]
dilation_lst = [(1, 1, 1)]
out_pad_lst = [(0, 0, 0)]
groups_lst = [1, 2]
dtype_lst = (
    [torch.float, torch.half, torch.bfloat16]
    if TEST_BFLOAT16
    else [torch.float, torch.half]
)


class TestConv3dOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_conv3d(self):
        channel_funcs = [self.convert_to_channel_last, lambda x: x, self.to_non_dense]
        product_list = product(
            bias_lst,
            N_lst,
            Ci_lst,
            D_lst,
            DHW_lst,
            Co_lst,
            kernel_lst,
            padding_lst,
            stride_lst,
            dilation_lst,
            groups_lst,
            channel_funcs,
            dtype_lst,
        )

        for (
            b,
            N,
            Ci,
            D,
            DHW,
            Co,
            K,
            pad,
            stride,
            dilation,
            groups,
            channel_func,
            dtype,
        ) in product_list:
            err = 0.003
            x = torch.randn(N, Ci, D, DHW, DHW, dtype=dtype, requires_grad=True)
            cm = nn.Conv3d(
                Ci,
                Co,
                K,
                stride=stride,
                bias=b,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ).to(dtype=dtype)
            cpu_cm = copy.deepcopy(cm).float()
            output_cpu = cpu_cm(x.float())
            grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
            if b:
                bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())
            x.grad.zero_()
            qcm = cm.mlu()
            output_mlu = qcm(channel_func(to_mlu(x)))
            output_mlu.backward(channel_func(to_mlu(grad_cpu)))
            x_grad_mlu = x.grad.contiguous().float()
            w_grad_mlu = qcm.weight.grad.cpu().contiguous().float()
            if b:
                # see [CONV bias grad Threshold adjustment]
                bias_err = 0.004 if dtype is torch.bfloat16 else err
                bias_grad_mlu = qcm.bias.grad.cpu().float()
                self.assertTensorsEqual(
                    bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
                )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().contiguous().float(), err, use_MSE=True
            )
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, err, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, err, use_MSE=True)
        # test double type separately
        b = True
        N = 8
        Ci = 8
        D = 18
        DHW = 20
        Co = 4
        K = (1, 2, 2)
        pad = 1
        stride = (1, 2, 2)
        dilation = (1, 1, 1)
        groups = 1
        channel_func = lambda x: x
        dtype = torch.double
        err = 0.003
        x = torch.randn(N, Ci, D, DHW, DHW, dtype=dtype, requires_grad=True)
        cm = nn.Conv3d(
            Ci,
            Co,
            K,
            stride=stride,
            bias=b,
            dilation=dilation,
            padding=pad,
            groups=groups,
        ).to(dtype=dtype)
        cpu_cm = copy.deepcopy(cm).float()
        output_cpu = cpu_cm(x.float())
        grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
        output_cpu.backward(grad_cpu)
        x_grad_cpu = copy.deepcopy(x.grad.float())
        w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
        if b:
            bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())
        x.grad.zero_()
        qcm = cm.mlu()
        output_mlu = qcm(channel_func(to_mlu(x)))
        output_mlu.backward(channel_func(to_mlu(grad_cpu)))
        x_grad_mlu = x.grad.contiguous().float()
        w_grad_mlu = qcm.weight.grad.cpu().contiguous().float()
        if b:
            # see [CONV bias grad Threshold adjustment]
            bias_err = 0.004 if dtype is torch.bfloat16 else err
            bias_grad_mlu = qcm.bias.grad.cpu().float()
            self.assertTensorsEqual(
                bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
            )
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().contiguous().float(), err, use_MSE=True
        )
        self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, err, use_MSE=True)
        self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_convtrans3d(self):
        channel_funcs = [self.convert_to_channel_last, lambda x: x, self.to_non_dense]
        product_list = product(
            bias_lst,
            N_lst,
            Ci_lst,
            D_lst,
            DHW_lst,
            Co_lst,
            kernel_lst,
            padding_lst,
            stride_lst,
            dilation_lst,
            out_pad_lst,
            groups_lst,
            channel_funcs,
            dtype_lst,
        )
        for (
            b,
            N,
            Ci,
            D,
            DHW,
            Co,
            K,
            _,
            stride,
            dilation,
            out_pad,
            groups,
            channel_func,
            dtype,
        ) in product_list:
            err = 0.003
            x = torch.randn(N, Ci, D, DHW, DHW, dtype=dtype, requires_grad=True)
            cm = nn.ConvTranspose3d(
                Ci,
                Co,
                K,
                stride=stride,
                output_padding=out_pad,
                bias=b,
                dilation=dilation,
                groups=groups,
            ).to(dtype=dtype)
            cpu_cm = copy.deepcopy(cm).float()
            output_cpu = cpu_cm(x.float())
            grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad_cpu)
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
            if b:
                bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())
            x.grad.zero_()
            qcm = cm.mlu()
            output_mlu = qcm(channel_func(to_mlu(x)))
            output_mlu.backward(channel_func(to_mlu(grad_cpu)))
            x_grad_mlu = x.grad.contiguous().float()
            w_grad_mlu = qcm.weight.grad.cpu().contiguous().float()
            if b:
                # see [CONV bias grad Threshold adjustment]
                bias_err = 0.004 if dtype is torch.bfloat16 else err
                bias_grad_mlu = qcm.bias.grad.cpu().float()
                self.assertTensorsEqual(
                    bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
                )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().contiguous().float(), err, use_MSE=True
            )
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, err, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, err, use_MSE=True)
            # test double type separately
        b = True
        N = 8
        Ci = 8
        D = 18
        DHW = 20
        Co = 4
        K = (1, 2, 2)
        pad = 1
        stride = (1, 2, 2)
        dilation = (1, 1, 1)
        groups = 1
        channel_func = lambda x: x
        dtype = torch.double
        err = 0.003
        x = torch.randn(N, Ci, D, DHW, DHW, dtype=dtype, requires_grad=True)
        cm = nn.Conv3d(
            Ci,
            Co,
            K,
            stride=stride,
            bias=b,
            dilation=dilation,
            padding=pad,
            groups=groups,
        ).to(dtype=dtype)
        cpu_cm = copy.deepcopy(cm).float()
        output_cpu = cpu_cm(x.float())
        grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
        output_cpu.backward(grad_cpu)
        x_grad_cpu = copy.deepcopy(x.grad.float())
        w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
        if b:
            bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())
        x.grad.zero_()
        qcm = cm.mlu()
        output_mlu = qcm(channel_func(to_mlu(x)))
        output_mlu.backward(channel_func(to_mlu(grad_cpu)))
        x_grad_mlu = x.grad.contiguous().float()
        w_grad_mlu = qcm.weight.grad.cpu().contiguous().float()
        if b:
            # see [CONV bias grad Threshold adjustment]
            bias_err = 0.004 if dtype is torch.bfloat16 else err
            bias_grad_mlu = qcm.bias.grad.cpu().float()
            self.assertTensorsEqual(
                bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
            )
        self.assertTensorsEqual(
            output_cpu, output_mlu.cpu().contiguous().float(), err, use_MSE=True
        )
        self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, err, use_MSE=True)
        self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, err, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv3d_exceptions(self):
        x = torch.randn(15)
        cm = nn.Conv3d(3, 5, 2)
        cm.to("mlu")
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = (
            "Expected 4D (unbatched) or 5D (batched) input to conv3d, "
            + "but got input of size: [15]"
        )
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(1, 7, 5, 5, 5)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = (
            "Given groups=1, weight of size [5, 3, 2, 2, 2], expected "
            + "input[1, 7, 5, 5, 5] to have 3 channels, but got 7 channels instead"
        )
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(10, 3, 5, 5, 5)
        x = x.to(torch.int8)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = "Convolution mlu op not implemented for 'Char'"
        self.assertEqual(info.exception.args[0], msg)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv3d_with_detach_tensor(self):
        device = "mlu"
        b = True
        N = 8
        Ci = 8
        D = 18
        DHW = 20
        Co = 4
        K = (1, 2, 2)
        pad = 1
        stride = (1, 2, 2)
        dilation = (1, 1, 1)
        groups = 1
        channel_func = lambda x: x
        dtype = torch.float
        err = 0.003
        cm = nn.Conv3d(
            Ci, Co, K, stride=stride_lst, padding=padding_lst, dilation=dilation_lst
        ).to(device)
        x = torch.rand(N, Ci, D, DHW, DHW, dtype=torch.float).to(device)
        with torch.no_grad():
            out = torch.nn.functional.conv3d(
                x,
                torch.randn_like(cm.weight).detach(),
                torch.randn_like(cm.bias).detach(),
            )
            out_no_detach = torch.nn.functional.conv3d(
                x, torch.randn_like(cm.weight), torch.randn_like(cm.bias)
            )
            message = "MLU Tensor Size and Detach Tensor are not equal !"
            self.assertEqual(out.size(), out_no_detach.size(), message)


if __name__ == "__main__":
    unittest.main()
