from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
import torch_mlu

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


class TestSmoothL1LossOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_forward(self):
        shape_list = [(1,), (2, 2), (3, 7, 8), (32, 4, 8732)]
        reduct_list = ["none", "sum", "mean"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True)
            target = torch.randn(item[0])

            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(
                self.to_mlu_dtype(x, item[2]), self.to_mlu_dtype(target, item[2])
            )
            self.assertTensorsEqual(
                out_cpu,
                out_mlu.cpu(),
                0.003,
                use_MSE=True,
                message=str(item[1] + " " + str(item[2])),
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_forward_not_dense(self):
        shape_list = [(32, 4, 5, 8732), (5, 3, 2, 3, 30)]
        reduct_list = ["none", "sum", "mean"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])[:, :, :, :15]
            target = torch.randn(item[0]).to(item[2])[:, :, :, :15]

            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(x.mlu(), target.mlu())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_forward_channels_last(self):
        shape_list = [(32, 4, 5, 8732), (5, 3, 2, 3, 10)]
        reduct_list = ["none", "sum", "mean"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            x = self.convert_to_channel_last(x)
            target = torch.randn(item[0]).to(item[2])
            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(x.mlu(), target.mlu())
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float().contiguous(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_backward_channel_last(self):
        shape_list = [(1,), (32, 4, 8732), (12, 3, 416, 416), (5, 3, 2, 3, 10)]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float]  # half is support for mlu
        channel_last = [False, True]
        for item in product(shape_list, reduct_list, dtype_list):
            for channel in channel_last:
                x = torch.randn(item[0], requires_grad=True).to(item[2])
                target = torch.randn(item[0]).to(item[2])
                layer = torch.nn.SmoothL1Loss(reduction=item[1])
                out_cpu = layer(x, target)
                grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
                grad_output_mlu = grad_output.to(torch.device("mlu"))
                out_cpu.backward(grad_output)
                grad_input_cpu = copy.deepcopy(x.grad)

                x.grad.zero_()
                layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
                y = x
                if channel is False:
                    y = self.convert_to_channel_last(x)
                out_mlu = layer_mlu(y.mlu(), target.mlu())
                out_mlu_ptr = out_mlu.data_ptr()
                out_mlu.backward(grad_output_mlu)
                grad_input_mlu = copy.deepcopy(x.grad)

                self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float().contiguous(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    grad_input_cpu, grad_input_mlu, 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_backward(self):
        shape_list = [
            (1,),
            (2, 2),
            (3, 7, 8),
            (32, 4, 8732),
            (12, 3, 416, 416),
            (5, 3, 2, 3, 10),
            (0, 2, 3),
        ]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            target = torch.randn(item[0]).to(item[2])
            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
            grad_output_mlu = grad_output.to(torch.device("mlu"))
            out_cpu.backward(grad_output)
            grad_input_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(x.mlu(), target.mlu())
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(grad_output_mlu)
            grad_input_mlu = copy.deepcopy(x.grad)

            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(grad_input_cpu, grad_input_mlu, 0.003, use_MSE=True)

            # not contiguous test
            if len(item[0]) in (3, 4):
                x_ori = torch.randn(item[0], requires_grad=True).to(item[2])
                target_ori = torch.randn(item[0]).to(item[2])
                x = x_ori[:, 1:2, 2:6]
                target = target_ori[:, 1:2, 2:6]
                layer = torch.nn.SmoothL1Loss(reduction=item[1])
                out_cpu = layer(x, target)
                grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
                grad_output_mlu = grad_output.to(torch.device("mlu"))
                out_cpu.backward(grad_output)
                grad_input_cpu = copy.deepcopy(x_ori.grad)

                x_ori.grad.zero_()
                layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
                out_mlu = layer_mlu(x.mlu(), target.mlu())
                out_mlu_ptr = out_mlu.data_ptr()
                out_mlu.backward(grad_output_mlu)
                grad_input_mlu = copy.deepcopy(x_ori.grad)

                self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )
                self.assertTensorsEqual(
                    grad_input_cpu, grad_input_mlu, 0.003, use_RAE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_PYTORCH_11152(self):
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(reduct_list, dtype_list):
            x = torch.randn((1, 4, 1, 64, 64)).to(item[1])
            target = torch.randn((1, 4, 1, 64, 64)).to(item[1])
            x.as_strided_(x.size(), stride=(4, 1, 4, 256, 4)).requires_grad_()
            target.as_strided_(target.size(), stride=(16384, 1, 16384, 256, 4))
            layer = torch.nn.SmoothL1Loss(reduction=item[0])
            out_cpu = layer(x, target)
            grad_output = torch.randn_like(out_cpu, dtype=torch.float)
            grad_output_mlu = grad_output.to(torch.device("mlu"))
            out_cpu.backward(grad_output)
            grad_input_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[0])
            out_mlu = layer_mlu(x.mlu(), target.mlu())
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(grad_output_mlu)
            grad_input_mlu = copy.deepcopy(x.grad)

            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)
            self.assertTensorsEqual(grad_input_cpu, grad_input_mlu, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_permute(self):
        reduct_list = ["none", "mean", "sum"]
        shape_list = [(3, 7, 8), (32, 4, 8732), (12, 3, 416, 416), (5, 3, 2, 3, 10)]
        permute_shape = [(0, 2, 1), (2, 1, 0), (0, 3, 2, 1), (0, 4, 3, 2, 1)]
        for i in range(4):
            for reduct in reduct_list:
                pm = permute_shape[i]
                x = torch.randn(shape_list[i])
                target = torch.randn(shape_list[i])
                x_mlu, target_mlu = x.to("mlu"), target.to("mlu")
                x, target = x.permute(pm), target.permute(pm)
                x_mlu, target_mlu = x_mlu.permute(pm), target_mlu.permute(pm)
                x.requires_grad = True
                x_mlu.requires_grad = True
                layer = torch.nn.SmoothL1Loss(reduction=reduct)
                output = layer(x, target)
                if reduct == "none":
                    grad_cpu = torch.ones(shape_list[i])
                    grad_mlu = grad_cpu.to("mlu").permute(pm)
                    grad_cpu = grad_cpu.permute(pm)
                else:
                    grad_cpu = torch.ones(output.shape)
                    grad_mlu = grad_cpu.to("mlu")
                output.backward(grad_cpu)
                grad_input = copy.deepcopy(x.grad)
                x.grad.zero_()
                output_mlu = layer(x_mlu, target_mlu)
                output_mlu.backward(grad_mlu)
                grad_input_mlu = copy.deepcopy(x_mlu.grad)
                self.assertTensorsEqual(output, output_mlu.cpu(), 0.003, use_MSE=True)
                self.assertTensorsEqual(
                    grad_input.float(),
                    grad_input_mlu.cpu().float(),
                    0.003,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_dtype(self):
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype, reduct in product(dtype_list, reduct_list):
            x = torch.randn((2, 3, 4, 5, 6)).to(torch.half).to(torch.float)
            target = torch.randn((2, 3, 4, 5, 6)).to(torch.half).to(torch.float)
            x_mlu, target_mlu = self.to_mlu_dtype(x, dtype), self.to_mlu_dtype(
                target, dtype
            )
            x.requires_grad = True
            x_mlu.requires_grad = True
            layer = torch.nn.SmoothL1Loss(reduction=reduct)
            output = layer(x, target)
            grad_cpu = torch.ones(output.shape)
            grad_mlu = self.to_mlu_dtype(grad_cpu, dtype)
            output.backward(grad_cpu)
            grad_input = copy.deepcopy(x.grad)
            x.grad.zero_()
            output_mlu = layer(x_mlu, target_mlu)
            self.assertTrue(dtype == output_mlu.dtype)
            output_mlu.backward(grad_mlu)
            grad_input_mlu = copy.deepcopy(x_mlu.grad)
            self.assertTensorsEqual(
                output, output_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTrue(dtype == grad_input_mlu.dtype)
            self.assertTensorsEqual(
                grad_input,
                grad_input_mlu.float().cpu(),
                0.003,
                use_MSE=True,
                message=str(dtype) + " " + str(reduct),
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("35GB")
    def test_smooth_l1_loss_forward_large(self):
        shape_list = [(5, 1024, 1024, 1024)]
        reduct_list = ["none"]
        dtype_list = [torch.half]
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            target = torch.randn(item[0]).to(item[2])

            layer = torch.nn.SmoothL1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.SmoothL1Loss(reduction=item[1])
            out_mlu = layer_mlu(
                self.to_mlu_dtype(x, item[2]), self.to_mlu_dtype(target, item[2])
            )
            self.assertTensorsEqual(
                out_cpu,
                out_mlu.cpu(),
                0.003,
                use_MSE=True,
                message=str(item[1] + " " + str(item[2])),
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_smooth_l1_loss_bfloat16(self):
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.bfloat16]
        for dtype, reduct in product(dtype_list, reduct_list):
            x = torch.randn((2, 3, 4, 5, 6), dtype=dtype)
            target = torch.randn((2, 3, 4, 5, 6), dtype=dtype)
            x_mlu, target_mlu = x.mlu(), target.mlu()
            x.requires_grad = True
            x_mlu.requires_grad = True
            layer = torch.nn.SmoothL1Loss(reduction=reduct)
            output = layer(x, target)
            grad_cpu = torch.ones(output.shape, dtype=dtype)
            grad_mlu = grad_cpu.to(dtype).mlu()
            output.backward(grad_cpu)
            grad_input = copy.deepcopy(x.grad)
            x.grad.zero_()
            output_mlu = layer(x_mlu, target_mlu)
            self.assertTrue(dtype == output_mlu.dtype)
            output_mlu.backward(grad_mlu)
            grad_input_mlu = copy.deepcopy(x_mlu.grad)
            self.assertTensorsEqual(
                output.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
            )
            self.assertTrue(dtype == grad_input_mlu.dtype)
            self.assertTensorsEqual(
                grad_input.float(),
                grad_input_mlu.float().cpu(),
                0.003,
                use_MSE=True,
                message=str(dtype) + " " + str(reduct),
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_smooth_l1_loss_beta(self):
        shape_list = [(1,), (2, 2), (3, 7, 8), (0, 2, 3), (32, 4, 8732)]
        reduct_list = ["none", "sum", "mean"]
        beta_list = [0.0, 0.1, 1.1, 5.8]
        for shape, reduction, beta in product(shape_list, reduct_list, beta_list):
            x = torch.randn(shape)
            target = torch.randn(shape)
            x_mlu = copy.deepcopy(x).mlu()
            x.requires_grad_(True)
            x_mlu.requires_grad_(True)
            target_mlu = copy.deepcopy(target).mlu()

            m = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
            out_cpu = m(x, target)
            out_mlu = m(x_mlu, target_mlu)
            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

            out_cpu.sum().backward()
            out_mlu.sum().backward()
            self.assertTensorsEqual(x.grad, x_mlu.grad.cpu(), 0.003, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
