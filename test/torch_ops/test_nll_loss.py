from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
from itertools import product

import torch
from torch import nn
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


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss1d_non_batch(self):
        C_lst = [4, 20, 1000]
        ignore_lst = [0, 1, -100]
        reduct_lst = ["none", "mean", "sum"]
        weight_lst = [True, False]
        # set half threshold to 3e-2 since this op can not fit 3e-3 when
        # input is half while mlu and gpu is consistent.
        dtype_err_lst = [(torch.float, 0), (torch.half, 3e-2), (torch.double, 0)]
        product_lst = product(reduct_lst, C_lst, ignore_lst, weight_lst, dtype_err_lst)
        for reduct, C, ignore, weight_flag, dtype_err in product_lst:
            dtype, err = dtype_err
            x = torch.randn(C, dtype=dtype)
            weight = torch.randn(C, dtype=dtype).abs()
            x_mlu = x.to("mlu")
            weight_mlu = weight.to("mlu")

            if dtype == torch.half:
                x = x.to(torch.float)
                weight = weight.to(torch.float)

            if not weight_flag:
                weight = None
                weight_mlu = None

            target = torch.randint(0, C, [], dtype=torch.long)

            layer = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.NLLLoss(
                weight_mlu, reduction=reduct, ignore_index=ignore
            )
            out_mlu = layer_mlu(x_mlu, self.to_device(target))
            # Special Case:
            # nll_loss1d: output[i] = - weight[i] * input[target[i]]
            # where i != ignore_index.
            # if target.item() == ignore and reduct == "mean",
            # gpu will return zero, while cpu and mlu return NaN (0/0).
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_nll_loss1d_non_batch_bfloat16(self):
        C_lst = [4, 20, 1000]
        ignore_lst = [0, 1, -100]
        reduct_lst = ["none", "mean", "sum"]
        weight_lst = [True, False]
        # set bfloat threshold to 3e-2 since this op can not fit 3e-3 when
        # input is bfloat while mlu and gpu is consistent.
        dtype_err_lst = [(torch.bfloat16, 3e-2)]
        product_lst = product(reduct_lst, C_lst, ignore_lst, weight_lst, dtype_err_lst)
        for reduct, C, ignore, weight_flag, dtype_err in product_lst:
            dtype, err = dtype_err
            x = torch.randn(C, dtype=dtype)
            weight = torch.randn(C, dtype=dtype).abs()
            x_mlu = x.to("mlu")
            weight_mlu = weight.to("mlu")

            if dtype == torch.bfloat16:
                x = x.to(torch.float)
                weight = weight.to(torch.float)

            if not weight_flag:
                weight = None
                weight_mlu = None

            target = torch.randint(0, C, [], dtype=torch.long)

            layer = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
            out_cpu = layer(x, target)
            layer_mlu = torch.nn.NLLLoss(
                weight_mlu, reduction=reduct, ignore_index=ignore
            )
            out_mlu = layer_mlu(x_mlu, self.to_device(target))
            # Special Case:
            # nll_loss1d: output[i] = - weight[i] * input[target[i]]
            # where i != ignore_index.
            # if target.item() == ignore and reduct == "mean",
            # gpu will return zero, while cpu and mlu return NaN (0/0).
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss1d_batch(self):
        N_lst = [8, 64, 128]
        C_lst = [20, 1000]
        ignore_lst = [0, 1, -100]
        reduct_lst = ["none", "mean", "sum"]
        weight_lst = [True, False]
        # set half threshold to 3e-2 since this op can not fit 3e-3 when
        # input is half while mlu and gpu is consistent.
        dtype_err_lst = [(torch.float, 0), (torch.half, 3e-2), (torch.double, 0)]
        product_lst = product(
            reduct_lst, N_lst, C_lst, ignore_lst, weight_lst, dtype_err_lst
        )
        for reduct, N, C, ignore, weight_flag, dtype_err in product_lst:
            dtype, err = dtype_err
            x = torch.randn(N, C, dtype=dtype)
            weight = torch.randn(C, dtype=dtype).abs()
            x_mlu = x.to("mlu")
            weight_mlu = weight.to("mlu")

            if dtype == torch.half:
                x = x.to(torch.float)
                weight = weight.to(torch.float)

            if not weight_flag:
                weight = None
                weight_mlu = None

            target = torch.randint(0, C, [N], dtype=torch.long)

            layer = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
            out_cpu = layer(x, target)

            layer_mlu = torch.nn.NLLLoss(
                weight_mlu, reduction=reduct, ignore_index=ignore
            )
            out_mlu = layer_mlu(x_mlu, self.to_device(target))

            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

            # non-contiguous
            target_ = torch.randint(0, C, [N, 2], dtype=torch.long)
            out_cpu2 = layer(x, target_[:, 0])
            out_mlu2 = layer_mlu(x_mlu, self.to_device(target_)[:, 0])
            self.assertTensorsEqual(
                out_cpu2.float(), out_mlu2.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss_backward(self):  # pylint: disable=R0912
        N_lst = [8, 64, 128]
        C_lst = [20, 1000]
        ignore_lst = [0, 1, -100]
        reduct_lst = ["none", "sum", "mean"]
        weight_lst = [True, False]
        # set half threshold to 3e-2 since this op can not fit 3e-3 when
        # input is half while mlu and gpu is consistent.
        dtype_err_lst = [(torch.float, 0), (torch.half, 3e-2), (torch.double, 0)]
        product_lst = product(
            reduct_lst, N_lst, C_lst, ignore_lst, weight_lst, dtype_err_lst
        )
        for reduct, N, C, ignore, weight_flag, dtype_err in product_lst:
            torch.manual_seed(1)
            dtype, err = dtype_err
            x_t = torch.randn(N, C, dtype=dtype)
            x_t.requires_grad = True
            x_mlu = copy.deepcopy(x_t)
            weight = torch.randn(C, dtype=dtype).abs()
            weight_mlu = weight.to("mlu")

            x = x_t
            if dtype == torch.half:
                x = x_t.to(torch.float)
                weight = weight.to(torch.float)

            if not weight_flag:
                weight = None
                weight_mlu = None

            # generate target
            target = torch.randint(0, C, [N], dtype=torch.long)

            layer = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
            out_cpu = layer(x, target)
            if dtype == torch.half:
                grad = torch.ones(out_cpu.shape).to(dtype).to(torch.float)
            else:
                grad = torch.ones(out_cpu.shape).to(dtype)
            out_cpu.backward(grad)

            layer_mlu = torch.nn.NLLLoss(
                weight_mlu, reduction=reduct, ignore_index=ignore
            )
            out_mlu = layer_mlu(self.to_mlu_dtype(x_mlu, dtype), target.mlu())
            out_mlu.backward(self.to_mlu_dtype(grad, dtype))

            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

            self.assertTensorsEqual(
                x_t.grad.float(), x_mlu.grad.cpu().float(), err, use_MSE=True
            )

            # non-contiguous
            x_t = torch.randn(N, C, dtype=dtype)
            weight = torch.randn(C, 2, dtype=dtype).abs()
            weight_mlu = weight.to("mlu")[:, 0]
            x_t.requires_grad = True
            x_mlu = copy.deepcopy(x_t)
            x = x_t
            if dtype == torch.half:
                x = x_t.to(torch.float)
                weight = weight.to(torch.float)[:, 0]
            else:
                weight = weight[:, 0]

            if not weight_flag:
                weight = None
                weight_mlu = None

            target_ = torch.randint(0, C, [N, 2], dtype=torch.long)

            layer2 = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
            out_cpu2 = layer2(x, target_[:, 0])
            if dtype == torch.half:
                grad = torch.ones(out_cpu2.shape).to(dtype).to(torch.float)
            else:
                grad = torch.ones(out_cpu2.shape).to(dtype)
            out_cpu2.backward(grad)

            layer_mlu2 = torch.nn.NLLLoss(
                weight_mlu, reduction=reduct, ignore_index=ignore
            )
            out_mlu2 = layer_mlu2(x_mlu.to("mlu"), self.to_device(target_)[:, 0])
            out_mlu2.backward(self.to_mlu_dtype(grad, dtype))

            self.assertTensorsEqual(
                out_cpu2.float(), out_mlu2.cpu().float(), err, use_MSE=True
            )

            self.assertTensorsEqual(
                x_t.grad.float(), x_mlu.grad.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_nll_loss_backward_bfloat16(self):  # pylint: disable=R0912
        N_lst = [8, 64, 128]
        C_lst = [20, 1000]
        ignore_lst = [0, 1, -100]
        reduct_lst = ["none", "sum", "mean"]
        weight_lst = [True, False]
        # TODO(guwei): Need to improve nll loss bp err theshord
        # set bfloat16 threshold to 3e-1 since this op can not fit 3e-3
        dtype_err_lst = [(torch.bfloat16, 3e-1)]
        product_lst = product(
            reduct_lst, N_lst, C_lst, ignore_lst, weight_lst, dtype_err_lst
        )
        for reduct, N, C, ignore, weight_flag, dtype_err in product_lst:
            dtype, err = dtype_err
            x_t = torch.randn(N, C, dtype=dtype)
            x_t.requires_grad = True
            x_mlu = copy.deepcopy(x_t)
            weight = torch.randn(C, dtype=dtype).abs()
            weight_mlu = weight.to("mlu")

            x = x_t
            if dtype == torch.bfloat16:
                x = x_t.to(torch.float)
                weight = weight.to(torch.float)

            if not weight_flag:
                weight = None
                weight_mlu = None

            # generate target
            target = torch.randint(0, C, [N], dtype=torch.long)

            layer = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
            out_cpu = layer(x, target)
            if dtype == torch.bfloat16:
                grad = torch.ones(out_cpu.shape).to(dtype).to(torch.float)
            else:
                grad = torch.ones(out_cpu.shape).to(dtype)
            out_cpu.backward(grad)

            layer_mlu = torch.nn.NLLLoss(
                weight_mlu, reduction=reduct, ignore_index=ignore
            )
            out_mlu = layer_mlu(self.to_mlu_dtype(x_mlu, dtype), target.mlu())
            out_mlu.backward(self.to_mlu_dtype(grad, dtype))

            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), err, use_MSE=True
            )

            self.assertTensorsEqual(
                x_t.grad.float(), x_mlu.grad.cpu().float(), err, use_MSE=True
            )

            # non-contiguous
            x_t = torch.randn(N, C, dtype=dtype)
            weight = torch.randn(C, 2, dtype=dtype).abs()
            weight_mlu = weight.to("mlu")[:, 0]
            x_t.requires_grad = True
            x_mlu = copy.deepcopy(x_t)
            x = x_t
            if dtype == torch.bfloat16:
                x = x_t.to(torch.float)
                weight = weight.to(torch.float)[:, 0]
            else:
                weight = weight[:, 0]

            if not weight_flag:
                weight = None
                weight_mlu = None

            target_ = torch.randint(0, C, [N, 2], dtype=torch.long)

            layer2 = torch.nn.NLLLoss(weight, reduction=reduct, ignore_index=ignore)
            out_cpu2 = layer2(x, target_[:, 0])
            if dtype == torch.bfloat16:
                grad = torch.ones(out_cpu2.shape).to(dtype).to(torch.float)
            else:
                grad = torch.ones(out_cpu2.shape).to(dtype)
            out_cpu2.backward(grad)

            layer_mlu2 = torch.nn.NLLLoss(
                weight_mlu, reduction=reduct, ignore_index=ignore
            )
            out_mlu2 = layer_mlu2(x_mlu.to("mlu"), self.to_device(target_)[:, 0])
            out_mlu2.backward(self.to_mlu_dtype(grad, dtype))

            self.assertTensorsEqual(
                out_cpu2.float(), out_mlu2.cpu().float(), err, use_MSE=True
            )

            self.assertTensorsEqual(
                x_t.grad.float(), x_mlu.grad.cpu().float(), err, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss_2d(self):
        N, C = 2, 3
        D = 4
        reduct_list = ["none", "sum", "mean"]
        for reduction in reduct_list:
            loss = nn.NLLLoss(reduction=reduction)
            input_t = torch.randn((N, C, D, D), dtype=torch.float, requires_grad=True)
            target = torch.empty(N, D, D, dtype=torch.long).random_(0, C)
            input_copy = copy.deepcopy(input_t)
            input_mlu = input_copy.to("mlu")
            target_mlu = target.to("mlu")
            output = loss(input_t, target)
            output_mlu = loss(input_mlu, target_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu(), 3e-3, use_MSE=True)

            # non-contiguous grad
            if reduction == "none":
                grad = torch.randn((N, D, D * 2), dtype=torch.float, requires_grad=True)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                output.backward(grad[:, :, :D])
                output_mlu.backward(grad_mlu[:, :, :D])
                self.assertTensorsEqual(
                    input_t.grad, input_copy.grad.cpu(), 3e-3, use_MSE=True
                )
            else:
                grad = torch.randn(output.shape, dtype=torch.float, requires_grad=True)
                grad_mlu = copy.deepcopy(grad).to("mlu")
                output.backward(grad)
                output_mlu.backward(grad_mlu)
                self.assertTensorsEqual(
                    input_t.grad, input_copy.grad.cpu(), 3e-3, use_MSE=True
                )

            # non-contiguous
            input_t_ = copy.deepcopy(input_t)

            input_copy_ = copy.deepcopy(input_t)
            input_mlu_ = input_copy_.to("mlu")
            target_copy = copy.deepcopy(target)

            output_ = loss(input_t_, target.transpose(1, 2).contiguous())
            output_mlu_ = loss(
                input_mlu_, target_copy.to("mlu").transpose(1, 2).contiguous()
            )
            self.assertTensorsEqual(output_, output_mlu_.cpu(), 3e-3, use_MSE=True)

            grad_ = torch.randn(output_.shape, dtype=torch.float, requires_grad=True)
            grad_mlu_ = copy.deepcopy(grad_).to("mlu")

            output_.backward(grad_)
            output_mlu_.backward(grad_mlu_)
            self.assertTensorsEqual(
                input_t_.grad, input_copy_.grad.cpu(), 3e-3, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss_2d_empty(self):
        n_list = [0, 3]
        h_list = [0, 5]
        w_list = [0, 5]
        reduction_list = ["none", "sum", "mean"]
        C = 3
        product_lst = product(n_list, h_list, w_list, reduction_list)
        for N, H, W, reduction in product_lst:
            loss = nn.NLLLoss(reduction=reduction)
            input_t = torch.randn((N, C, H, W), dtype=torch.float, requires_grad=True)
            target = torch.empty(N, H, W, dtype=torch.long).random_(0, C)
            input_copy = copy.deepcopy(input_t)
            input_mlu = input_copy.to("mlu")
            target_mlu = target.to("mlu")
            output = loss(input_t, target)
            output_mlu = loss(input_mlu, target_mlu)
            err = 0
            mse_flag = False
            if N != 0 and H != 0 and W != 0:
                err = 0.003
                mse_flag = True
            self.assertTensorsEqual(output, output_mlu.cpu(), err, use_MSE=mse_flag)

            grad = torch.randn(output.shape, dtype=torch.float, requires_grad=True)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            output.backward(grad)
            output_mlu.backward(grad_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu(), err, use_MSE=mse_flag)

    # @unittest.skip("not test")
    @testinfo()
    def test_nll_loss_exception(self):
        loss = nn.NLLLoss()
        input = torch.randn((10, 4), dtype=torch.float, requires_grad=True).to("mlu")
        target = torch.empty((10, 5), dtype=torch.long).random_(0, 4).to("mlu")
        ref_msg = r"1D target tensor expected, multi-target not supported"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            loss(input, target)

        loss = nn.NLLLoss()
        input = torch.randn((10, 4), dtype=torch.float, requires_grad=True).to("mlu")
        target = torch.empty(9, dtype=torch.long).random_(0, 4).to("mlu")
        ref_msg = r"Expected input batch_size \(10\) to match target batch_size \(9\)\."
        with self.assertRaisesRegex(ValueError, ref_msg):
            loss(input, target)

        loss = nn.NLLLoss(weight=torch.randn(5, dtype=torch.float).to("mlu"))
        input = torch.randn((10, 4), dtype=torch.float, requires_grad=True).to("mlu")
        target = torch.empty((10), dtype=torch.long).random_(0, 4).to("mlu")
        ref_msg = (
            r"weight tensor should be defined either for all 4 classes or no classes"
        )
        ref_msg = ref_msg + r" but got weight tensor of shape: \[5\]"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            loss(input, target)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("38GB")
    def test_nll_loss_2d_large(self):
        N, C = 48, 13725
        D = 64
        reduct_list = ["sum"]
        for reduction in reduct_list:
            loss = nn.NLLLoss(reduction=reduction)
            input_t = torch.randn((N, C, D, D), dtype=torch.float, requires_grad=True)
            target = torch.empty(N, D, D, dtype=torch.long).random_(0, C)
            input_copy = copy.deepcopy(input_t)
            input_mlu = input_copy.to("mlu")
            target_mlu = target.to("mlu")
            output = loss(input_t, target)
            output_mlu = loss(input_mlu, target_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu(), 3e-3, use_MSE=True)

            # non-contiguous grad
            grad = torch.randn(output.shape, dtype=torch.float, requires_grad=True)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            output.backward(grad)
            output_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                input_t.grad, input_copy.grad.cpu(), 3e-3, use_MSE=True
            )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("38GB")
    def test_nll_loss_large(self):
        N, C = 2**16, 2**16 + 1
        reduct_list = ["sum"]
        for reduction in reduct_list:
            loss = nn.NLLLoss(reduction=reduction)
            input_t = torch.randn((N, C), dtype=torch.float, requires_grad=True)
            target = torch.empty(N, dtype=torch.long).random_(0, C)
            input_copy = copy.deepcopy(input_t)
            input_mlu = input_copy.to("mlu")
            target_mlu = target.to("mlu")
            output = loss(input_t, target)
            output_mlu = loss(input_mlu, target_mlu)
            self.assertTensorsEqual(output, output_mlu.cpu(), 3e-3, use_MSE=True)

            # non-contiguous grad
            grad = torch.randn(output.shape, dtype=torch.float, requires_grad=True)
            grad_mlu = copy.deepcopy(grad).to("mlu")
            output.backward(grad)
            output_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                input_t.grad, input_copy.grad.cpu(), 3e-3, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
