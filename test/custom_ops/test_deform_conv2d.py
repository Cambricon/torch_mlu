from __future__ import print_function
import os
import copy
from itertools import product  # pylint: disable=W0611
import logging
import sys
import unittest
import random
import torch
from torchvision.ops import deform_conv2d

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
logging.basicConfig(level=logging.DEBUG)
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411


class TestDeformConv2dOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_deform_conv2d(self):
        input_shapes = [(10, 15, 128, 128), (5, 15, 160, 160)]
        kernel_sizes = [(3, 3), (3, 2)]
        strides = [(1, 1), (1, 2)]
        paddings = [(1, 1), (3, 2)]
        dilations = [(1, 1), (2, 1)]
        groups = [1, 2]
        offset_groups = [1, 2]
        product_list = product(
            input_shapes,
            kernel_sizes,
            strides,
            paddings,
            dilations,
            groups,
            offset_groups,
        )
        for (
            shape,
            k_size,
            stride,
            padding,
            dilation,
            group,
            offset_group,
        ) in product_list:
            use_mask = True if random.randint(0, 1) else False
            use_bias = True if random.randint(0, 1) else False
            bs, in_c, in_h, in_w = shape
            if in_c % offset_group or in_c % group:
                continue
            kh, kw = k_size
            stride_h, stride_w = stride
            pad_h, pad_w = padding
            dilation_h, dilation_w = dilation
            ker_h = int(dilation_h * (kh - 1) + 1)
            ker_w = int(dilation_w * (kw - 1) + 1)
            if (in_h + 2 * pad_h - ker_h) % stride_h:
                continue
            if (in_w + 2 * pad_w - ker_w) % stride_w:
                continue
            out_h = int(((in_h + 2 * pad_h - ker_h) / stride_h) + 1)
            out_w = int(((in_w + 2 * pad_w - ker_w) / stride_w) + 1)
            w1 = int(in_c / group)
            w0 = int(group * random.randint(1, 15))

            input = torch.randn(shape, dtype=torch.float, requires_grad=True)
            weight_shape = (w0, w1, kh, kw)
            weight = torch.randn(weight_shape, dtype=torch.float, requires_grad=True)
            offset_shape = (bs, 2 * offset_group * kh * kw, out_h, out_w)
            offset = torch.randn(offset_shape, dtype=torch.float, requires_grad=True)
            mask = None
            if use_mask:
                mask_shpe = (bs, offset_group * kh * kw, out_h, out_w)
                mask = torch.randn(mask_shpe, dtype=torch.float, requires_grad=True)
            bias = None
            if use_bias:
                bias = torch.randn(w0, dtype=torch.float, requires_grad=True)

            input_ = copy.deepcopy(input)
            weight_ = copy.deepcopy(weight)
            offset_ = copy.deepcopy(offset)
            mask_ = copy.deepcopy(mask) if use_mask else None
            bias_ = copy.deepcopy(bias) if use_bias else None

            input_mlu = self.to_mlu(input_)
            weight_mlu = self.to_mlu(weight_)
            offset_mlu = self.to_mlu(offset_)
            mask_mlu = self.to_mlu(mask_) if use_mask else None
            bias_mlu = self.to_mlu(bias_) if use_bias else None

            out_cpu = deform_conv2d(
                input,
                offset,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                mask=mask,
            )
            out_mlu = deform_conv2d(
                input_mlu,
                offset_mlu,
                weight_mlu,
                bias=bias_mlu,
                stride=stride,
                padding=padding,
                dilation=dilation,
                mask=mask_mlu,
            )

            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu = self.to_mlu(copy.deepcopy(grad))

            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input.grad, input_.grad, 0.003, use_MSE=True)
            self.assertTensorsEqual(weight.grad, weight_.grad, 0.003, use_MSE=True)
            self.assertTensorsEqual(offset.grad, offset_.grad, 0.003, use_MSE=True)
            if use_mask:
                self.assertTensorsEqual(mask.grad, mask_.grad, 0.003, use_MSE=True)
            if use_bias:
                self.assertTensorsEqual(bias.grad, bias_.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_deform_conv2d_channels_last_and_no_dense(self):
        func_list = [self.convert_to_channel_last, self.to_non_dense]
        input_shape = (10, 15, 128, 128)
        weight_shape = (15, 15, 3, 3)
        offset_shape = (10, 18, 126, 126)
        mask_shape = (10, 9, 126, 126)
        bias_shape = 15

        for func in func_list:
            input = torch.randn(input_shape, dtype=torch.float, requires_grad=True)
            weight = torch.randn(weight_shape, dtype=torch.float, requires_grad=True)
            offset = torch.randn(offset_shape, dtype=torch.float, requires_grad=True)
            mask = torch.randn(mask_shape, dtype=torch.float, requires_grad=True)
            bias = torch.randn(bias_shape, dtype=torch.float, requires_grad=True)

            input_ = copy.deepcopy(input)
            weight_ = copy.deepcopy(weight)
            offset_ = copy.deepcopy(offset)
            mask_ = copy.deepcopy(mask)
            bias_ = copy.deepcopy(bias)

            input_mlu = self.to_mlu(input_)
            weight_mlu = self.to_mlu(weight_)
            offset_mlu = self.to_mlu(offset_)
            mask_mlu = self.to_mlu(mask_)
            bias_mlu = self.to_mlu(bias_)

            out_cpu = deform_conv2d(
                func(input),
                func(offset),
                func(weight),
                bias=func(bias),
                mask=func(mask),
            )
            out_mlu = deform_conv2d(
                func(input_mlu),
                func(offset_mlu),
                func(weight_mlu),
                bias=func(bias_mlu),
                mask=func(mask_mlu),
            )

            grad = torch.randn(out_cpu.shape, dtype=torch.float)
            grad_mlu = self.to_mlu(copy.deepcopy(grad))

            out_cpu.backward(grad)
            out_mlu.backward(grad_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(input.grad, input_.grad, 0.003, use_MSE=True)
            self.assertTensorsEqual(weight.grad, weight_.grad, 0.003, use_MSE=True)
            self.assertTensorsEqual(offset.grad, offset_.grad, 0.003, use_MSE=True)
            self.assertTensorsEqual(mask.grad, mask_.grad, 0.003, use_MSE=True)
            self.assertTensorsEqual(bias.grad, bias_.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_deform_conv2d_type(self):
        dtype_list = [torch.float]  # , torch.half]  please ref CNNLCORE-9995
        input_shape = (5, 15, 160, 160)
        weight_shape = (15, 15, 3, 3)
        offset_shape = (5, 18, 158, 158)
        mask_shape = (5, 9, 158, 158)
        bias_shape = 15

        for type in dtype_list:
            input = torch.randn(input_shape, dtype=type, requires_grad=True)
            weight = torch.randn(weight_shape, dtype=type, requires_grad=True)
            offset = torch.randn(offset_shape, dtype=type, requires_grad=True)
            mask = torch.randn(mask_shape, dtype=type, requires_grad=True)
            bias = torch.randn(bias_shape, dtype=type, requires_grad=True)

            input_ = copy.deepcopy(input)
            weight_ = copy.deepcopy(weight)
            offset_ = copy.deepcopy(offset)
            mask_ = copy.deepcopy(mask)
            bias_ = copy.deepcopy(bias)

            input_mlu = self.to_mlu(input_)
            weight_mlu = self.to_mlu(weight_)
            offset_mlu = self.to_mlu(offset_)
            mask_mlu = self.to_mlu(mask_)
            bias_mlu = self.to_mlu(bias_)

            out_cpu = deform_conv2d(
                input.float(),
                offset.float(),
                weight.float(),
                bias=bias.float(),
                mask=mask.float(),
            )
            out_mlu = deform_conv2d(
                input_mlu, offset_mlu, weight_mlu, bias=bias_mlu, mask=mask_mlu
            )

            grad = torch.randn(out_cpu.shape, dtype=type)
            grad_mlu = self.to_mlu(copy.deepcopy(grad))

            out_cpu.backward(grad.float())
            out_mlu.backward(grad_mlu)

            self.assertTensorsEqual(out_cpu, out_mlu.float().cpu(), 0.003, use_MSE=True)
            self.assertTensorsEqual(
                input.grad.float(), input_.grad.float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                weight.grad.float(), weight_.grad.float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                offset.grad.float(), offset_.grad.float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                mask.grad.float(), mask_.grad.float(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                bias.grad.float(), bias_.grad.float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_deform_conv2d_spec(self):
        input_shape = (0, 15, 160, 160)
        weight_shape = (15, 15, 3, 3)
        offset_shape = (0, 18, 158, 158)
        mask_shape = (0, 9, 158, 158)
        bias_shape = 15

        input = torch.randn(input_shape, dtype=torch.float, requires_grad=True)
        weight = torch.randn(weight_shape, dtype=torch.float, requires_grad=True)
        offset = torch.randn(offset_shape, dtype=torch.float, requires_grad=True)
        mask = torch.randn(mask_shape, dtype=torch.float, requires_grad=True)
        bias = torch.randn(bias_shape, dtype=torch.float, requires_grad=True)

        input_ = copy.deepcopy(input)
        weight_ = copy.deepcopy(weight)
        offset_ = copy.deepcopy(offset)
        mask_ = copy.deepcopy(mask)
        bias_ = copy.deepcopy(bias)

        input_mlu = self.to_mlu(input_)
        weight_mlu = self.to_mlu(weight_)
        offset_mlu = self.to_mlu(offset_)
        mask_mlu = self.to_mlu(mask_)
        bias_mlu = self.to_mlu(bias_)

        out_cpu = deform_conv2d(input, offset, weight, bias=bias, mask=mask)
        out_mlu = deform_conv2d(
            input_mlu, offset_mlu, weight_mlu, bias=bias_mlu, mask=mask_mlu
        )

        grad = torch.randn(out_cpu.shape, dtype=torch.float)
        grad_mlu = self.to_mlu(copy.deepcopy(grad))

        out_cpu.backward(grad)
        out_mlu.backward(grad_mlu)

        self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
        self.assertTensorsEqual(input.grad, input_.grad, 0.003, use_MSE=True)
        self.assertTensorsEqual(weight.grad, weight_.grad, 0.003, use_MSE=True)
        self.assertTensorsEqual(offset.grad, offset_.grad, 0.003, use_MSE=True)
        self.assertTensorsEqual(mask.grad, mask_.grad, 0.003, use_MSE=True)
        self.assertTensorsEqual(bias.grad, bias_.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_deform_conv2d_with_amp(self):
        bias = torch.randn(256)
        input = torch.randn(2, 256, 116, 200)
        mask = torch.randn(2, 9, 116, 200)
        offset = torch.randn(2, 18, 116, 200)
        weight = torch.randn(256, 256, 3, 3)
        with torch.autocast(enabled=True, device_type="mlu"):
            # autocast is not supported by CPU
            out_cpu = deform_conv2d(
                input, offset, weight, bias, stride=(1, 1), padding=(1, 1), mask=mask
            ).half()

            out_mlu = deform_conv2d(
                input.half().mlu(),
                offset.half().mlu(),
                weight.half().mlu(),
                bias.mlu(),
                stride=(1, 1),
                padding=(1, 1),
                mask=mask.mlu(),
            )
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
