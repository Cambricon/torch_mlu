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
    run_tests,
    TestCase,
    read_card_info,
    skipBFloat16IfNotSupport,
    TEST_LARGETENSOR,
    largeTensorTest,
)

TEST_BFLOAT16 = read_card_info()


def to_mlu(tensor_cpu):
    return tensor_cpu.mlu()


# Note [CONV bias grad Threshold adjustment]
# Now torch_mlu using cpu float conv result to diff2 with mlu bfloat16 / half conv result.
# For MLU bfloat16 conv test, torch_mlu using different threshold value to do this check.
# Bias grad is calculated by grad.sum((0, 2, 3)), or almost equal to this formula.
# Sum op get bfloat16 input, and using float as acc type, then convert to bfloat16 output.
# on-chip float type to off-chip bfloat16 loss precision, so this maybe failed by using
# fixed threshold 0.003. Now torch_mlu using 0.004 to avoid this failed.
# Also GPU result is same with MLU result when bfloat16 in this failed case:
# N, Ci, HW, K, P, S, D, dtype, Cout, seed :  1 3 7 3 1 1 1 torch.bfloat16 1708672435.3382535
# in test_depthwise_online_conv

dtype_lst = (
    [torch.float, torch.half, torch.double, torch.bfloat16]
    if TEST_BFLOAT16
    else [torch.float, torch.half, torch.double]
)


class TestConvOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_online_convtrans2d(self):
        bias_lst = [True, False]
        N_lst = [3]
        Ci_lst = [8]
        HW_lst = [24]
        Co_lst = [4]
        K_lst = [3]
        padding_lst = [0, 3]
        stride_lst = [2, 3]
        dilation_lst = [2, 3]
        output_padding_lst = [0, 1]
        groups_lst = [1, 2, 4]
        channel_func_lst = [
            self.convert_to_channel_last,
            lambda x: x,
            self.to_non_dense,
        ]
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
            x = torch.randn(N, Ci, HW, HW, dtype=dtype, requires_grad=True)
            w = torch.randn(Ci, Co // groups, K, K, dtype=dtype)
            if bias_t:
                bias = torch.randn(Co, dtype=dtype)
            cm = nn.ConvTranspose2d(
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
            cpu_cm = copy.deepcopy(cm).float()
            cpu_cm.weight = torch.nn.Parameter(w.float())
            if bias_t:
                cpu_cm.bias = torch.nn.Parameter(bias.float())
            output_cpu = cpu_cm(x.float())
            grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad_cpu.float())
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
            if bias_t:
                bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())
            x.grad.zero_()
            qcm = cm.mlu()
            qcm.weight = torch.nn.Parameter(channel_func(w.mlu()))
            if bias_t:
                qcm.bias = torch.nn.Parameter(channel_func(bias.mlu()))
            output_mlu = qcm(channel_func(x.mlu()))
            output_mlu.backward(channel_func(grad_cpu.mlu()))
            x_grad_mlu = x.grad.contiguous().float()
            w_grad_mlu = qcm.weight.grad.cpu().contiguous().float()
            if bias_t:
                # see [CONV bias grad Threshold adjustment]
                bias_err = 0.004 if dtype is torch.bfloat16 else er
                bias_grad_mlu = qcm.bias.grad.cpu().float()
                self.assertTensorsEqual(
                    bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
                )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().contiguous(), er, use_MSE=True
            )
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_online_conv2d(self):
        bias_lst = [True, False]
        N_lst = [32]
        Ci_lst = [32, 64]
        HW_lst = [14, 24]
        Co_lst = [64]
        K_lst = [2, 3]
        padding_lst = [0, 3]
        stride_lst = [1, 3]
        dilation_lst = [1]
        groups_lst = [1, 2, 4]
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
            dtype,
        ) in product_list:
            er = 0.003
            x = torch.randn(N, Ci, HW, HW, dtype=dtype, requires_grad=True)
            cm = nn.Conv2d(
                Ci,
                Co,
                K,
                bias=bias_t,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ).to(dtype=dtype)
            cpu_cm = copy.deepcopy(cm).float()
            output_cpu = cpu_cm(x.float())
            grad = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad.float())
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
            if bias_t:
                bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())
            x.grad.zero_()
            cm.mlu()
            output_mlu = cm(to_mlu(x))
            output_mlu.backward(to_mlu(grad))
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

    # Ref pytorch/test/test_nn.py:test_Conv2d_naive_groups,test_Conv2d_groups_nobias,
    #                             test_Conv2d_groups_nobias_v2
    # @unittest.skip("not test")
    @testinfo()
    def test_online_conv_groups(self):
        params_group = [
            [4, 4, 2, 2, False],
            [4, 4, 2, 2, True],
            [4, 16, 2, 8, False],
        ]
        for Ci, Co, ci, co, bias_t in params_group:
            i = torch.randn(2, Ci, 6, 6, requires_grad=True)
            w = torch.randn(Co, int(Ci / 2), 3, 3, requires_grad=True)
            if bias_t:
                bias = torch.randn(Co, requires_grad=True)
            qcm = nn.Conv2d(Ci, Co, 3, groups=2, bias=bias_t).float().mlu()
            qcm.weight = torch.nn.Parameter(w.to("mlu"))
            if bias_t:
                qcm.bias = torch.nn.Parameter(bias.to("mlu"))
            output = qcm(i.to("mlu"))
            grad_output = torch.randn(2, Co, 4, 4)
            output.backward(grad_output.to("mlu"))

            qcm1 = nn.Conv2d(ci, co, 3, bias=bias_t).float().mlu()
            qcm1.weight = torch.nn.Parameter(w[:co].to("mlu"))
            if bias_t:
                qcm1.bias = torch.nn.Parameter(bias[:co].to("mlu"))
            i1 = i.data[:, :ci].contiguous().requires_grad_(True)
            output1 = qcm1(i1.to("mlu"))
            output1.backward(grad_output[:, :co].contiguous().to("mlu"))

            qcm2 = nn.Conv2d(ci, co, 3, bias=bias_t).float().mlu()
            qcm2.weight = torch.nn.Parameter(w[co:].to("mlu"))
            if bias_t:
                qcm2.bias = torch.nn.Parameter(bias[co:].to("mlu"))
            i2 = i.data[:, ci:].contiguous().requires_grad_(True)
            output2 = qcm2(i2.to("mlu"))
            output2.backward(grad_output[:, co:].contiguous().to("mlu"))

            self.assertEqual(output.cpu(), torch.cat([output1.cpu(), output2.cpu()], 1))
            self.assertEqual(
                i.grad, torch.cat([i1.grad, i2.grad], 1), atol=1e-5, rtol=0
            )
            if bias_t:
                self.assertEqual(
                    qcm.bias.grad.cpu(),
                    torch.cat([qcm1.bias.grad.cpu(), qcm2.bias.grad.cpu()], 0),
                    atol=1e-5,
                    rtol=0,
                )
            self.assertEqual(
                qcm.weight.grad.cpu(),
                torch.cat([qcm1.weight.grad.cpu(), qcm2.weight.grad.cpu()], 0),
                atol=1e-5,
                rtol=0,
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_depthwise_online_conv(self):
        N_lst = [1, 8, 32, 64]
        Ci_lst = [6, 16]
        HW_lst = [12, 24]
        K_lst = [3]
        padding = [1]
        stride = [1]
        dilation = [1, 2]
        loop_var = [N_lst, Ci_lst, HW_lst, K_lst, padding, stride, dilation, dtype_lst]
        for N, Ci, HW, K, P, S, D, dtype in product(*loop_var):
            m = (1, 2, 10)
            Cout = (Ci * x for x in m)
            for Co in Cout:
                err = 0.003
                x = torch.rand(N, Ci, HW, HW, dtype=dtype, requires_grad=True)
                cm = nn.Conv2d(
                    Ci, Co, K, bias=True, stride=S, padding=P, dilation=D, groups=Ci
                ).to(dtype=dtype)
                cpu_cm = copy.deepcopy(cm).float()
                output_cpu = cpu_cm(x.float())
                grad = torch.randn(output_cpu.shape, dtype=dtype)
                output_cpu.backward(grad.float())
                x_grad_cpu = copy.deepcopy(x.grad.float())
                w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
                bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())

                x.grad.zero_()
                cm.mlu()
                output_mlu = cm(to_mlu(x))
                output_mlu.backward(to_mlu(grad))
                x_grad_mlu = x.grad.cpu().float()
                w_grad_mlu = cm.weight.grad.cpu().float()
                bias_grad_mlu = cm.bias.grad.cpu().float()
                self.assertTensorsEqual(
                    output_cpu, output_mlu.cpu().float(), err, use_MSE=True
                )
                self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, err, use_MSE=True)
                self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, err, use_MSE=True)
                # see [CONV bias grad Threshold adjustment]
                bias_err = 0.004 if dtype is torch.bfloat16 else err
                self.assertTensorsEqual(
                    bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_online_depthwise_convtrans2d(self):
        bias_lst = [False, True]
        N_lst = [16]
        Ci_lst = [64]
        HW_lst = [32]
        Co_lst = [64]
        K_lst = [8]
        padding_lst = [2]
        stride_lst = [4]
        dilation_lst = [1, 2]
        output_padding_lst = [0]
        groups_lst = [64]
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
            x = torch.randn(N, Ci, HW, HW, dtype=dtype, requires_grad=True)
            cm = nn.ConvTranspose2d(
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
            cpu_cm = copy.deepcopy(cm).float()
            output_cpu = cpu_cm(x.float())
            grad_cpu = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad_cpu.float())
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
            if bias_t:
                bias_grad_cpu = copy.deepcopy(cpu_cm.bias.grad.float())
            x.grad.zero_()
            qcm = cm.mlu()
            output_mlu = qcm(to_mlu(channel_func(x)))
            output_mlu.backward(to_mlu(channel_func(grad_cpu)))
            x_grad_mlu = x.grad.contiguous().float()
            w_grad_mlu = qcm.weight.grad.cpu().contiguous().float()
            if bias_t:
                # see [CONV bias grad Threshold adjustment]
                bias_err = 0.004 if dtype is torch.bfloat16 else er
                bias_grad_mlu = qcm.bias.grad.cpu().float()
                self.assertTensorsEqual(
                    bias_grad_cpu, bias_grad_mlu, bias_err, use_MSE=True
                )
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().contiguous().float(), er, use_MSE=True
            )
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv2d_exceptions(self):
        x = torch.randn(15)
        cm = nn.Conv2d(3, 5, 2)
        cm.to("mlu")
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [15]"
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(1, 7, 5, 3)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = (
            "Given groups=1, weight of size [5, 3, 2, 2], expected input[1, 7, 5, 3] "
            + "to have 3 channels, but got 7 channels instead"
        )
        self.assertEqual(info.exception.args[0], msg)

        x = torch.randn(10, 3, 5, 5)
        x = x.to(torch.int)
        with self.assertRaises(RuntimeError) as info:
            _ = cm(x.to("mlu"))
        msg = "Input type (int) and bias type (float) should be the same"
        self.assertEqual(info.exception.args[0], msg)

    # @unittest.skip("not test")
    @testinfo()
    def test_conv2d_with_detach_tensor(self):
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
            cm = nn.Conv2d(
                Ci, Co, K, stride=stride, padding=padding, dilation=dilation
            ).to(device)
            x = torch.rand(N, Ci, HW, HW, dtype=torch.float).to(device)
            with torch.no_grad():
                out = torch.nn.functional.conv2d(
                    x,
                    torch.randn_like(cm.weight).detach(),
                    torch.randn_like(cm.bias).detach(),
                )
                out_no_detach = torch.nn.functional.conv2d(
                    x, torch.randn_like(cm.weight), torch.randn_like(cm.bias)
                )
                message = "MLU Tensor Size and Detach Tensor are not equal !"
                self.assertEqual(out.size(), out_no_detach.size(), message)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("34GB")
    def test_conv2d_large(self):
        bias_t = False
        N = 4 * 1025
        Ci = 1024
        HW = 32
        Co = 1024
        K = 4
        padding = 0
        stride = 1
        dilation = 1
        groups = 1
        dtype = torch.half
        er = 0.003
        x = torch.randn(N, Ci, HW, HW, dtype=dtype)
        cm = nn.Conv2d(
            Ci,
            Co,
            K,
            bias=bias_t,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        ).to(dtype=dtype)
        cpu_cm = copy.deepcopy(cm).float()
        output_cpu = cpu_cm(x.float())
        cm.mlu()
        output_mlu = cm(to_mlu(x))
        self.assertTensorsEqual(output_cpu, output_mlu.cpu().float(), er, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("64GB")
    def test_conv2d_bp_large(self):
        # [CNNL] [Error]:[cnnlConvolutionBackwardData] overflow max supported tensor num 2147483647,
        # now tensor's total num is 4299161600.
        ref_msg = "CNNL error: CNNL_STATUS_NOT_SUPPORTED"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            bias_t = False
            N = 4 * 1025
            Ci = 1024
            HW = 32
            Co = 1024
            K = 4
            padding = 0
            stride = 1
            dilation = 1
            groups = 1
            dtype = torch.half
            er = 0.003
            x = torch.randn(N, Ci, HW, HW, dtype=dtype, requires_grad=True)
            cm = nn.Conv2d(
                Ci,
                Co,
                K,
                bias=bias_t,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ).to(dtype=dtype)
            cpu_cm = copy.deepcopy(cm).float()
            output_cpu = cpu_cm(x.float())
            grad = torch.randn(output_cpu.shape, dtype=dtype)
            output_cpu.backward(grad.float())
            x_grad_cpu = copy.deepcopy(x.grad.float())
            w_grad_cpu = copy.deepcopy(cpu_cm.weight.grad.float())
            x.grad.zero_()
            cm.mlu()
            output_mlu = cm(to_mlu(x))
            output_mlu.backward(to_mlu(grad))
            x_grad_mlu = x.grad.cpu().float()
            w_grad_mlu = cm.weight.grad.cpu().float()
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu().float(), er, use_MSE=True
            )
            self.assertTensorsEqual(x_grad_cpu, x_grad_mlu, er, use_MSE=True)
            self.assertTensorsEqual(w_grad_cpu, w_grad_mlu, er, use_MSE=True)


if __name__ == "__main__":
    run_tests()
