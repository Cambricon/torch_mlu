from __future__ import print_function
import logging
import sys
import os
import copy
from itertools import product
import unittest

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


dtype_lst = (
    [torch.float, torch.half, torch.double, torch.bfloat16]
    if TEST_BFLOAT16
    else [torch.float, torch.half, torch.double]
)


class TestConvOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_convtrans1d(self):
        bias_lst = [True, False]
        N_lst = [3]
        Ci_lst = [64, 32]
        HW_lst = [20, 11]
        Co_lst = [32, 16]
        K_lst = [1, 2]
        padding_lst = [1, 3]
        stride_lst = [1, 3]
        dilation_lst = [1, 2]
        output_padding_lst = [0]
        groups_lst = [1, 2, 4]
        channel_func_lst = [self.convert_to_channel_last, lambda x: x]
        product_list = product(
            bias_lst,
            N_lst,
            Ci_lst,
            HW_lst,
            Co_lst,
            K_lst,
            padding_lst,
            stride_lst,
            dilation_lst,
            output_padding_lst,
            groups_lst,
            channel_func_lst,
            dtype_lst,
        )
        for (
            bias_t,
            N,
            Ci,
            HW,
            Co,
            K,
            padding,
            stride,
            dilation,
            output_padding,
            groups,
            channel_func,
            dtype,
        ) in product_list:
            er = 0.003
            x = torch.randn(N, Ci, HW, dtype=dtype, requires_grad=True)
            cm = nn.ConvTranspose1d(
                Ci,
                Co,
                K,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias_t,
                dilation=dilation,
                groups=groups,
            ).to(dtype=dtype)
            cm_cpu = copy.deepcopy(cm).float()
            output_cpu = cm_cpu(x.float())
            grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad_cpu.float())
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cm_cpu.weight.grad.float())
            if bias_t:
                bias_grad_cpu = copy.deepcopy(cm_cpu.bias.grad.float())
            x.grad.zero_()
            qcm = cm.mlu()
            output_mlu = qcm(channel_func(to_mlu(x)))
            output_mlu.backward(channel_func(to_mlu(grad_cpu)))
            x_grad_mlu = x.grad.cpu().float()
            w_grad_mlu = qcm.weight.grad.cpu().float()
            if bias_t:
                # see [CONV bias grad Threshold adjustment]
                bias_err = 0.004 if dtype is torch.bfloat16 else er
                bias_grad_mlu = qcm.bias.grad.cpu().float()
                self.assertTensorsEqual(
                    bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
                )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), er, use_MSE=True
            )
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv1d(self):
        bias_lst = [True, False]
        N_lst = [1, 32]
        Ci_lst = [128, 256]
        HW_lst = [11, 20]
        Co_lst = [64, 80, 128]
        K_lst = [1, 2]
        padding_lst = [0, 3]
        stride_lst = [1, 3]
        dilation_lst = [1, 2]
        groups_lst = [1, 2, 4]
        channel_func_lst = [self.convert_to_channel_last, lambda x: x]
        product_list = product(
            bias_lst,
            N_lst,
            Ci_lst,
            HW_lst,
            Co_lst,
            K_lst,
            padding_lst,
            stride_lst,
            dilation_lst,
            groups_lst,
            channel_func_lst,
            dtype_lst,
        )
        for (
            bias_t,
            N,
            Ci,
            HW,
            Co,
            K,
            padding,
            stride,
            dilation,
            groups,
            channel_func,
            dtype,
        ) in product_list:
            x = torch.randn(N, Ci, HW, dtype=dtype, requires_grad=True)
            er = 0.003
            cm = nn.Conv1d(
                Ci,
                Co,
                K,
                bias=bias_t,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ).to(dtype=dtype)
            cm_cpu = copy.deepcopy(cm).float()
            output_cpu = cm_cpu(x.float())
            grad = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad.float())
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cm_cpu.weight.grad.float())
            if bias_t:
                bias_grad_cpu = copy.deepcopy(cm_cpu.bias.grad.float())
            x.grad.zero_()
            cm.mlu()
            output_mlu = cm(channel_func(to_mlu(x)))
            output_mlu.backward(channel_func(to_mlu(grad)))
            x_grad_mlu = x.grad.cpu().float()
            w_grad_mlu = cm.weight.grad.cpu().float()
            if bias_t:
                # see [CONV bias grad Threshold adjustment]
                bias_err = 0.004 if dtype is torch.bfloat16 else er
                bias_grad_mlu = cm.bias.grad.cpu().float()
                self.assertTensorsEqual(
                    bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
                )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), er, use_MSE=True
            )
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv1d_exceptions(self):
        x = torch.randn(15)
        cm = nn.Conv1d(3, 5, 2)
        cm.to("mlu")
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = "Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [15]"
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(1, 7, 5)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = (
            "Given groups=1, weight of size [5, 3, 2], expected input[1, 7, 5] "
            + "to have 3 channels, but got 7 channels instead"
        )

        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(10, 3, 5)
        x = x.to(torch.uint8)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = "Convolution mlu op not implemented for 'Byte'"
        self.assertEqual(info.exception.args[0], msg)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv1d_with_detach_tensor(self):
        device = "mlu"
        N_lst = [20]
        Ci_lst = [16]
        HW_lst = [50, 100]
        Co_lst = [33]
        K_lst = [3, 3]
        padding_lst = [0]
        stride_lst = [1, 3]
        dilation_lst = [1, 2]
        product_list = product(
            N_lst, Ci_lst, HW_lst, Co_lst, K_lst, padding_lst, stride_lst, dilation_lst
        )
        for N, Ci, HW, Co, K, padding, stride, dilation in product_list:
            cm = nn.Conv1d(
                Ci, Co, K, stride=stride, padding=padding, dilation=dilation
            ).to(device)
            x = torch.rand(N, Ci, HW, dtype=torch.float).to(device)
            with torch.no_grad():
                out = torch.nn.functional.conv1d(
                    x,
                    torch.randn_like(cm.weight).detach(),
                    torch.randn_like(cm.bias).detach(),
                )
                out_no_detach = torch.nn.functional.conv1d(
                    x, torch.randn_like(cm.weight), torch.randn_like(cm.bias)
                )
                message = "MLU Tensor Size and Detach Tensor are not equal !"
                self.assertEqual(out.size(), out_no_detach.size(), message)


if __name__ == "__main__":
    unittest.main()
