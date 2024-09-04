from __future__ import print_function

import sys
import os
import unittest
import logging

import torch
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class ConvCat(torch.nn.Module):
    r"""
    Conv2d + view + cat
    """

    def __init__(self):
        super(ConvCat, self).__init__()
        self.cm = torch.nn.Conv2d(384, 128, 1, bias=False, stride=1, padding=0)

    def set_weight(self, weight):
        self.cm.weight = weight

    def forward(self, input):
        y = self.cm(input)
        output = torch.cat([y.view(-1), input.view(-1)])
        return output


def get_tensor_cl_stride(input: torch.Tensor, replace_stride_value=False):
    r"""
    Caculate cl stride based on tensor size.
    """

    flag = input.is_contiguous(
        memory_format=torch.channels_last
    ) or input.is_contiguous(memory_format=torch.channels_last_3d)
    assert flag, "Only support cl stride caculate."
    dim = input.dim()
    result = []
    if dim == 4:
        stride1 = 1
        stride3 = (
            input.size()[1]
            if (input.size()[3] != 1 or replace_stride_value is False)
            else 1
        )
        stride2 = stride3 * input.size()[3]
        stride0 = stride2 * input.size()[2] if stride2 != 1 else input.size()[1]
        result = [stride0, stride1, stride2, stride3]
    else:
        stride1 = 1
        stride4 = (
            input.size()[1]
            if (input.size()[4] != 1 or replace_stride_value is False)
            else 1
        )
        stride3 = stride4 * input.size()[4]
        stride2 = stride3 * input.size()[3]
        stride0 = stride2 * input.size()[2] if stride2 != 1 else input.size()[1]
        result = [stride0, stride1, stride2, stride3, stride4]
    return tuple(result)


def get_tensor_stride(input: torch.Tensor, replace_stride_value=False):
    r"""
    Get tensor stride based on tensor size.
    """
    assert input.device.type == "mlu", "Only support mlu device tensor."
    if input.is_contiguous(memory_format=torch.channels_last) or input.is_contiguous(
        memory_format=torch.channels_last_3d
    ):
        return get_tensor_cl_stride(input, replace_stride_value)
    return input.stride()


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_view(self):
        for in_shape, out_shape in [
            ((1, 1000, 1, 1), (1, -1)),
            ((1, 3, 200, 200), (1, -1, 1, 200, 200)),
            ((1,), (1,)),
            ((1, 58, 2, 28, 28), (2, -1, 4)),
            ((45, 54, 454), (45, 54, 454)),
        ]:
            x_cpu = torch.randint(0, 100, (in_shape))
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu.view(out_shape)
            y_mlu = x_mlu.view(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # This test case is used to test whether view op can handle 64 bit input correctly when
    # the input is smaller than max int32/float32 (complex128 will not be tested since CNNL
    # does not support cast complex input)
    # @unittest.skip("not test")
    @testinfo()
    def test_view_64_bit(self):
        for in_shape, out_shape in [
            ((1, 1000, 1, 1), (1, -1)),
            ((1, 3, 200, 200), (1, -1, 1, 200, 200)),
            ((1,), (1,)),
            ((1, 58, 2, 28, 28), (2, -1, 4)),
            ((45, 54, 454), (45, 54, 454)),
        ]:
            for dtype in [torch.int64, torch.double]:
                x_cpu = torch.randn(in_shape).to(dtype)
                x_mlu = self.to_device(x_cpu)
                y_cpu = x_cpu.view(out_shape)
                y_mlu = x_mlu.view(out_shape)
                self.assertTrue(y_cpu.size() == y_mlu.size())
                self.assertTrue(y_cpu.stride() == y_mlu.stride())
                self.assertTensorsEqual(y_cpu, y_mlu.cpu().float(), 0)
                self.assertTrue(x_cpu.size() == x_mlu.size())
                self.assertTrue(x_cpu.stride() == x_mlu.stride())
                self.assertTensorsEqual(x_cpu, x_mlu.cpu(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_channels_last_and_not_dense(self):
        for in_shape, out_shape in [
            ((1, 1000, 1, 1), (1, -1)),
            ((2, 3, 4, 5), (2, -1, 1, 4, 5)),
        ]:
            x_cpu = torch.randn(in_shape).to(memory_format=torch.channels_last)
            x_mlu = self.to_device(x_cpu)
            y_cpu = x_cpu[:, :2].view(out_shape)
            y_mlu = x_mlu[:, :2].view(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_channels_last_unsafe(self):
        for in_shape, out_shape in [
            ((64, 3, 24, 24), (1, -1)),
            ((2, 3, 4, 5), (-1, 5)),
            ((13, 77, 23, 153), (23, 1001, 153)),
            ((13, 77, 23, 153, 3), (23, 1001, 153, 3)),
        ]:
            x_cpu = torch.randn(in_shape)
            x_cl = self.convert_to_channel_last(x_cpu)
            x_mlu = self.to_device(x_cl)
            y_cpu = x_cpu.view(out_shape)
            y_mlu = x_mlu.view(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(
                get_tensor_stride(x_mlu) == x_mlu.stride()
                or get_tensor_stride(x_mlu, True) == x_mlu.stride()
            )
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)

    @unittest.skip("not test")
    @testinfo()
    def test_conv2d_and_view(self):
        func_list = [lambda x: x, self.to_non_dense, self.convert_to_channel_last]
        er = 0.003
        for func in func_list:
            x = torch.nn.Parameter(torch.randn(48, 384, 20, 16, dtype=torch.float))
            w = torch.nn.Parameter(torch.randn(128, 384, 1, 1, dtype=torch.float))
            x_mlu = torch.nn.Parameter(func(x.clone().detach().mlu()))
            w_mlu = torch.nn.Parameter(func(w.clone().detach().mlu()))
            net = ConvCat()
            net_mlu = ConvCat().to("mlu")
            net.set_weight(w)
            net_mlu.set_weight(w_mlu)
            # test cpu side
            output_cpu = net(x)
            # generate grad
            grad = torch.randn(output_cpu.shape, dtype=torch.float)
            grad_mlu = func(grad.mlu())
            # test cpu backward
            output_cpu.backward(grad)
            # test mlu forward and backward
            output_mlu = net_mlu(x_mlu)
            output_mlu.backward(grad_mlu)
            self.assertTrue(x.grad.size() == x_mlu.grad.size())
            self.assertTrue(
                get_tensor_stride(x_mlu.grad) == x_mlu.grad.stride()
                or get_tensor_stride(x_mlu.grad, True) == x_mlu.grad.stride()
            )
            self.assertTrue(w.grad.size() == w_mlu.grad.size())
            self.assertTrue(
                get_tensor_stride(w_mlu.grad) == w_mlu.grad.stride()
                or get_tensor_stride(w_mlu.grad, True) == w_mlu.grad.stride()
            )
            self.assertTrue(output_cpu.size() == output_mlu.size())
            self.assertTrue(
                get_tensor_stride(output_mlu) == output_mlu.stride()
                or get_tensor_stride(output_mlu, True) == output_mlu.stride()
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), er, use_MSE=True)
            self.assertTensorsEqual(x.grad, x_mlu.grad.cpu(), er, use_MSE=True)
            self.assertTensorsEqual(w.grad, w_mlu.grad.cpu(), er, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_dtype(self):
        # half -> int16
        x = torch.randn(5, 5, dtype=torch.half)
        x_mlu = x.to("mlu")
        out_mlu = x_mlu.view(torch.int16)
        base_cpu = x.numpy().view(np.int16)
        self.assertEqual(out_mlu.cpu(), base_cpu, 0)

        # float -> int32
        x = torch.randn(5, 5, dtype=torch.float)
        x_mlu = x.to("mlu")
        out_mlu = x_mlu.view(torch.int32)
        base_cpu = x.numpy().view(np.int32)
        self.assertEqual(out_mlu.cpu(), base_cpu, 0)

    # @unittest.skip("not test")
    @testinfo()
    def test_view_device(self):
        if torch.mlu.device_count() <= 1:
            return
        for in_shape, out_shape in [
            ((1, 1000, 1, 1), (1, -1)),
            ((0,), (0,)),
            ((2, 3, 4, 0), (2, 3, 4, 0)),
        ]:
            device = "mlu:1"
            x_cpu = torch.randint(0, 100, (in_shape))
            x_cpu_cl = self.convert_to_channel_last(x_cpu)
            x_mlu = x_cpu.mlu(device)
            x_mlu_cl = self.convert_to_channel_last(x_mlu)
            torch.mlu.set_device(0)
            y_cpu = x_cpu_cl.view(out_shape)
            y_mlu = x_mlu_cl.view(out_shape)
            self.assertTrue(y_cpu.size() == y_mlu.size())
            self.assertTrue(y_cpu.stride() == y_mlu.stride())
            self.assertTensorsEqual(y_cpu.float(), y_mlu.cpu().float(), 0)
            self.assertTrue(x_cpu.size() == x_mlu.size())
            self.assertTrue(x_cpu.stride() == x_mlu.stride())
            self.assertTensorsEqual(x_cpu.float(), x_mlu.cpu().float(), 0)
            self.assertTrue(y_mlu.device == x_mlu_cl.device)
            self.assertTrue(y_mlu.device == torch.device(device))


if __name__ == "__main__":
    unittest.main()
