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

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_l1_loss_forward(self):
        shape_list = [(1,), (2, 2), (32, 4, 8732), (12, 3, 416, 416), (5, 3, 2, 3, 10)]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True, dtype=torch.float)
            target = torch.randn(item[0], dtype=torch.float)

            layer = torch.nn.L1Loss(reduction=item[1])
            out_cpu = layer(x, target)

            layer_mlu = torch.nn.L1Loss(reduction=item[1])
            out_mlu = layer_mlu(x.to("mlu").to(item[2]), target.to("mlu").to(item[2]))

            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_l1_loss_forward_not_dense(self):
        shape_list = [(32, 4, 8732), (12, 3, 416, 416)]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x_ori = torch.randn(item[0], requires_grad=True).to(torch.float)
            target_ori = torch.randn(item[0]).to(torch.float)
            x = x_ori[:, 1:2, 2:6]
            target = target_ori[:, 1:2, 2:6]
            layer = torch.nn.L1Loss(reduction=item[1])
            out_cpu = layer(x, target)

            layer_mlu = torch.nn.L1Loss(reduction=item[1])
            out_mlu = layer_mlu(
                self.to_mlu_dtype(x, item[2]), self.to_mlu_dtype(target, item[2])
            )

            self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_l1_loss_forward_channels_last(self):
        shape_list = [(32, 4, 5, 8732), (5, 3, 2, 3, 10)]
        reduct_list = ["none", "sum", "mean"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True).to(item[2])
            x = self.convert_to_channel_last(x)
            target = torch.randn(item[0]).to(item[2])
            layer = torch.nn.L1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.L1Loss(reduction=item[1])
            out_mlu = layer_mlu(self.to_device(x), target.mlu())
            self.assertTensorsEqual(
                out_cpu, out_mlu.cpu().float().contiguous(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_l1_loss_backward(self):
        shape_list = [
            (1, 2),
            (2, 2),
            (3, 7, 8),
            (32, 4, 8732),
            (12, 3, 416, 416),
            (5, 3, 2, 3, 10),
        ]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float, torch.half]
        for item in product(shape_list, reduct_list, dtype_list):
            x = torch.randn(item[0], requires_grad=True, dtype=item[2])
            target = torch.randn(item[0], dtype=item[2])

            layer = torch.nn.L1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
            grad_output_mlu = grad_output.to(torch.device("mlu"))
            out_cpu.backward(grad_output)

            grad_input_cpu = copy.deepcopy(x.grad)

            x.grad.zero_()
            layer_mlu = torch.nn.L1Loss(reduction=item[1])
            out_mlu = layer_mlu(self.to_mlu(x), self.to_mlu(target))
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(grad_output_mlu)
            grad_input_mlu = copy.deepcopy(x.grad)

            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            if item[2] == torch.half:
                self.assertEqual(grad_input_cpu, grad_input_mlu.cpu())
            else:
                self.assertTensorsEqual(
                    grad_input_cpu.float(),
                    grad_input_mlu.cpu().float(),
                    0.003,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_l1_loss_backward_not_dense(self):
        shape_list = [(32, 4, 8732), (12, 3, 416, 416)]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float]  # half is support for mlu
        for item in product(shape_list, reduct_list, dtype_list):
            x_ori = torch.randn(item[0], requires_grad=True).to(torch.float)
            target_ori = torch.randn(item[0]).to(torch.float)
            x = x_ori[:, 1:2, 2:6]
            target = target_ori[:, 1:2, 2:6]
            layer = torch.nn.L1Loss(reduction=item[1])
            out_cpu = layer(x, target)
            grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
            grad_output_mlu = grad_output.to(torch.device("mlu"))
            out_cpu.backward(grad_output)
            grad_input_cpu = copy.deepcopy(x_ori.grad)

            x_ori.grad.zero_()
            layer_mlu = torch.nn.L1Loss(reduction=item[1])
            out_mlu = layer_mlu(
                self.to_mlu_dtype(x, item[2]), self.to_mlu_dtype(target, item[2])
            )
            out_mlu_ptr = out_mlu.data_ptr()
            out_mlu.backward(grad_output_mlu)
            grad_input_mlu = copy.deepcopy(x_ori.grad)

            self.assertEqual(out_mlu_ptr, out_mlu.data_ptr())
            if item[2] == torch.half:
                self.assertTensorsEqual(
                    grad_input_cpu.float(),
                    grad_input_mlu.cpu().float(),
                    0.09,
                    use_MSE=True,
                )
            else:
                self.assertTensorsEqual(
                    grad_input_cpu.float(),
                    grad_input_mlu.cpu().float(),
                    0.003,
                    use_MSE=True,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_l1_loss_backward_channel_last(self):
        shape_list = [(1,), (32, 4, 8732), (12, 3, 416, 416), (5, 3, 2, 3, 10)]
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.float]  # half is support for mlu
        channel_last = [False, True]
        for item in product(shape_list, reduct_list, dtype_list):
            for channel in channel_last:
                x = torch.randn(item[0], requires_grad=True).to(item[2])
                target = torch.randn(item[0]).to(item[2])
                layer = torch.nn.L1Loss(reduction=item[1])
                out_cpu = layer(x, target)
                grad_output = torch.ones(out_cpu.shape, dtype=torch.float)
                grad_output_mlu = grad_output.to(torch.device("mlu"))
                out_cpu.backward(grad_output)
                grad_input_cpu = copy.deepcopy(x.grad)

                x.grad.zero_()
                layer_mlu = torch.nn.L1Loss(reduction=item[1])
                y = x
                if channel is False:
                    y = self.convert_to_channel_last(x)
                out_mlu = layer_mlu(self.to_device(y), target.mlu())
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
    def test_l1_loss_permute(self):
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
                layer = torch.nn.L1Loss(reduction=reduct)
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
                    message=str(shape_list[i]) + " " + str(reduct),
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_l1_loss_dtype(self):
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.double, torch.float, torch.half]
        for dtype, reduct in product(dtype_list, reduct_list):
            x = torch.randn((2, 3, 4, 5, 6), dtype=dtype)
            target = torch.randn((2, 3, 4, 5, 6), dtype=dtype)
            x_mlu, target_mlu = self.to_mlu(x), self.to_mlu(target)
            x.requires_grad = True
            x_mlu.requires_grad = True
            layer = torch.nn.L1Loss(reduction=reduct)
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
            self.assertTrue(dtype == grad_input_mlu.dtype)
            self.assertTensorsEqual(
                output.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
            )
            if dtype == torch.half:
                self.assertEqual(
                    grad_input, grad_input_mlu.cpu(), msg=str(dtype) + " " + str(reduct)
                )
            else:
                self.assertTensorsEqual(
                    grad_input.float(),
                    grad_input_mlu.float().cpu(),
                    0.003,
                    use_MSE=True,
                    message=str(dtype) + " " + str(reduct),
                )


if __name__ == "__main__":
    unittest.main()
