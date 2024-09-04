from __future__ import print_function

import sys
import os
import copy
from itertools import product
import unittest
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
    run_tests,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestBceOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_bce(self):
        shape_list = [(156), (2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3), (torch.double, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(
            shape_list, reduct_lst, dtype_list, weight_flag_list
        ):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight_orig = torch.rand(shape, dtype=torch.float).to(type_err[0])
            if weight_flag:
                weight_ = weight_orig
                weight_mlu = weight_orig.to("mlu")
            else:
                weight_ = None
                weight_mlu = None
            loss = nn.BCELoss(weight=weight_ if weight_flag else None, reduction=reduct)
            loss_mlu = nn.BCELoss(
                weight=weight_mlu if weight_flag else None, reduction=reduct
            )
            out_cpu = loss(x, target)
            out_mlu = loss_mlu(x.to("mlu"), target.to("mlu"))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_PYTORCH_11152(self):
        shape_list = [(1, 4, 1, 64, 64)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(
            shape_list, reduct_lst, dtype_list, weight_flag_list
        ):
            x = (
                torch.rand(shape, dtype=torch.float)
                .to(type_err[0])
                .as_strided_(shape, stride=(4, 1, 4, 256, 4))
                .requires_grad_()
            )
            target = (
                torch.rand(shape, dtype=torch.float)
                .to(type_err[0])
                .as_strided_(shape, stride=(16384, 1, 16384, 256, 4))
            )
            weight_orig = (
                torch.rand(shape, dtype=torch.float)
                .to(type_err[0])
                .as_strided_(target.size(), stride=(16384, 1, 16384, 256, 4))
            )
            if weight_flag:
                weight_ = weight_orig
                weight_mlu = weight_orig.to("mlu")
            else:
                weight_ = None
                weight_mlu = None
            loss = nn.BCELoss(weight=weight_ if weight_flag else None, reduction=reduct)
            loss_mlu = nn.BCELoss(
                weight=weight_mlu if weight_flag else None, reduction=reduct
            )
            out_cpu = loss(x, target)
            out_mlu = loss_mlu(x.to("mlu"), target.to("mlu"))
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
            )

            grad_in = torch.randn_like(out_cpu)
            grad_in_mlu = grad_in.to("mlu")
            if reduct == "none":
                out_cpu.backward(grad_in)
            else:
                out_cpu.backward()
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            if reduct == "none":
                out_mlu.backward(grad_in_mlu)
            else:
                out_mlu.backward()
            grad_mlu = copy.deepcopy(x.grad)
            x.grad.zero_()
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), type_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_not_dense(self):
        shape_list = [(2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3), (torch.double, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(
            shape_list, reduct_lst, dtype_list, weight_flag_list
        ):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight_orig = torch.rand(shape, dtype=torch.float).to(type_err[0])
            if weight_flag:
                weight_cpu = weight_orig[..., : int(shape[-1] / 2)]
                weight_mlu = weight_orig.to("mlu")[..., : int(shape[-1] / 2)]
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x[..., : int(shape[-1] / 2)]
            x_mlu = x.to("mlu")[..., : int(shape[-1] / 2)]
            target_cpu = target[..., : int(shape[-1] / 2)]
            target_mlu = target.to("mlu")[..., : int(shape[-1] / 2)]
            loss_cpu = nn.BCELoss(
                weight=weight_cpu if weight_flag else None, reduction=reduct
            )
            loss_mlu = nn.BCELoss(
                weight=weight_mlu if weight_flag else None, reduction=reduct
            )
            out_cpu = loss_cpu(x_cpu, target_cpu)
            out_mlu = loss_mlu(x_mlu, target_mlu)
            try:
                self.assertTensorsEqual(
                    out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
                )
            except AssertionError as e:
                print(e)

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_channel_last(self):
        shape_list = [(2, 4, 6, 8), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3), (torch.double, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(
            shape_list, reduct_lst, dtype_list, weight_flag_list
        ):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight_orig = torch.rand(shape, dtype=torch.float).to(type_err[0])
            if weight_flag:
                weight_cpu = weight_orig
                weight_mlu = weight_orig.to("mlu")
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x.to(memory_format=torch.channels_last)
            x_mlu = x.to("mlu").to(memory_format=torch.channels_last)
            target_cpu = target
            target_mlu = target.to("mlu")
            loss_cpu = nn.BCELoss(
                weight=weight_cpu if weight_flag else None, reduction=reduct
            )
            loss_mlu = nn.BCELoss(
                weight=weight_mlu if weight_flag else None, reduction=reduct
            )
            out_cpu = loss_cpu(x_cpu, target_cpu)
            out_mlu = loss_mlu(x_mlu, target_mlu)

            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_bp(self):
        shape_list = [(156), (2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(
            shape_list, reduct_lst, dtype_list, weight_flag_list
        ):
            x = torch.rand(shape, dtype=torch.float, requires_grad=True).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in_mlu = grad_in.to("mlu")
            if weight_flag:
                weight_ = weight
                weight_mlu = weight.to("mlu")
            else:
                weight_ = None
                weight_mlu = None
            out_cpu = F.binary_cross_entropy(
                x, target, reduction=reduct, weight=weight_
            )
            if reduct == "none":
                out_cpu.backward(grad_in)
            else:
                out_cpu.backward()
            grad_cpu = copy.deepcopy(x.grad)
            x.grad.zero_()
            out_mlu = F.binary_cross_entropy(
                x.to("mlu"), target.to("mlu"), reduction=reduct, weight=weight_mlu
            )
            if reduct == "none":
                out_mlu.backward(grad_in_mlu)
            else:
                out_mlu.backward()
            grad_mlu = copy.deepcopy(x.grad)
            x.grad.zero_()
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), type_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_bp_not_dense(self):
        shape_list = [(2, 4, 6, 8), (527, 80), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(
            shape_list, reduct_lst, dtype_list, weight_flag_list
        ):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in = torch.rand(shape, dtype=torch.float).to(type_err[0])[
                ..., : int(shape[-1] / 2)
            ]
            grad_in_mlu = grad_in.to("mlu")[..., : int(shape[-1] / 2)]
            if weight_flag:
                weight_cpu = weight[..., : int(shape[-1] / 2)]
                weight_mlu = weight.to("mlu")[..., : int(shape[-1] / 2)]
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x[..., : int(shape[-1] / 2)].requires_grad_()
            x_mlu = x.to("mlu")[..., : int(shape[-1] / 2)].requires_grad_()

            target_cpu = target[..., : int(shape[-1] / 2)]
            target_mlu = target.to("mlu")[..., : int(shape[-1] / 2)]
            out_cpu = F.binary_cross_entropy(
                x_cpu, target_cpu, reduction=reduct, weight=weight_cpu
            )
            if reduct == "none":
                out_cpu.backward(grad_in)
            else:
                out_cpu.backward()
            grad_cpu = copy.deepcopy(x_cpu.grad)
            x_cpu.grad.zero_()
            out_mlu = F.binary_cross_entropy(
                x_mlu, target_mlu, reduction=reduct, weight=weight_mlu
            )
            if reduct == "none":
                out_mlu.backward(grad_in_mlu)
            else:
                out_mlu.backward()
            grad_mlu = copy.deepcopy(x_mlu.grad)
            x_mlu.grad.zero_()
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), type_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_bp_channel_last(self):
        shape_list = [(2, 4, 6, 8), (32, 3, 14, 26)]
        reduct_lst = ["none", "mean", "sum"]
        dtype_list = [(torch.float, 3e-3)]
        weight_flag_list = [True, False]
        for shape, reduct, type_err, weight_flag in product(
            shape_list, reduct_lst, dtype_list, weight_flag_list
        ):
            x = torch.rand(shape, dtype=torch.float).to(type_err[0])
            target = torch.rand(shape, dtype=torch.float).to(type_err[0])
            weight = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in = torch.rand(shape, dtype=torch.float).to(type_err[0])
            grad_in_mlu = grad_in.to("mlu")
            if weight_flag:
                weight_cpu = weight
                weight_mlu = weight.to("mlu")
            else:
                weight_cpu = None
                weight_mlu = None
            x_cpu = x.to(memory_format=torch.channels_last).requires_grad_()
            x_mlu = x.to("mlu").to(memory_format=torch.channels_last).requires_grad_()
            target_cpu = target
            target_mlu = target.to("mlu")
            out_cpu = F.binary_cross_entropy(
                x_cpu, target_cpu, reduction=reduct, weight=weight_cpu
            )
            if reduct == "none":
                out_cpu.backward(grad_in)
            else:
                out_cpu.backward()
            grad_cpu = copy.deepcopy(x_cpu.grad)
            x_cpu.grad.zero_()
            out_mlu = F.binary_cross_entropy(
                x_mlu, target_mlu, reduction=reduct, weight=weight_mlu
            )
            if reduct == "none":
                out_mlu.backward(grad_in_mlu)
            else:
                out_mlu.backward()
            grad_mlu = copy.deepcopy(x_mlu.grad)
            x_mlu.grad.zero_()
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
            )
            self.assertTensorsEqual(
                grad_cpu.float(), grad_mlu.cpu().float(), type_err[1], use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_permute(self):
        reduct_list = ["none", "mean", "sum"]
        shape_list = [(3, 7, 8), (32, 4, 8732), (12, 3, 416, 416), (5, 3, 2, 3, 10)]
        permute_shape = [(0, 2, 1), (2, 1, 0), (0, 3, 2, 1), (0, 4, 3, 2, 1)]
        weight_flag_list = [True, False]
        for i in range(4):
            for reduct in reduct_list:
                for weight_flag in weight_flag_list:
                    pm = permute_shape[i]
                    x = torch.rand(shape_list[i])
                    target = torch.rand(shape_list[i])
                    weight = torch.rand(shape_list[i])
                    if weight_flag:
                        weight_cpu = weight
                        weight_mlu = weight.to("mlu").permute(pm)
                        weight_cpu = weight.permute(pm)
                    else:
                        weight_cpu = None
                        weight_mlu = None
                    x_mlu, target_mlu = x.to("mlu"), target.to("mlu")
                    x, target = x.permute(pm), target.permute(pm)
                    x_mlu, target_mlu = x_mlu.permute(pm), target_mlu.permute(pm)
                    x.requires_grad = True
                    x_mlu.requires_grad = True
                    output = F.binary_cross_entropy(
                        x, target, reduction=reduct, weight=weight_cpu
                    )
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
                    output_mlu = F.binary_cross_entropy(
                        x_mlu, target_mlu, reduction=reduct, weight=weight_mlu
                    )
                    output_mlu.backward(grad_mlu)
                    grad_input_mlu = copy.deepcopy(x_mlu.grad)
                    self.assertTensorsEqual(
                        output, output_mlu.cpu(), 0.003, use_MSE=True
                    )
                    self.assertTensorsEqual(
                        grad_input.float(),
                        grad_input_mlu.cpu().float(),
                        0.003,
                        use_MSE=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_bceloss_dtype(self):
        reduct_list = ["none", "mean", "sum"]
        dtype_list = [torch.double, torch.float, torch.half]
        weight_flag_list = [True, False]
        for dtype, reduct, weight_flag in product(
            dtype_list, reduct_list, weight_flag_list
        ):
            # for some cases, mlu half dtype may produce "inf" results, leading to NOT EQUAL with cpu float results
            torch.manual_seed(42)
            x = torch.rand((2, 3, 4, 5, 8), dtype=dtype).to(torch.float)
            target = torch.rand((2, 3, 4, 5, 8), dtype=dtype).to(torch.float)
            weight = torch.rand((2, 3, 4, 5, 8), dtype=dtype).to(torch.float)
            x_mlu, target_mlu = self.to_mlu_dtype(x, dtype), self.to_mlu_dtype(
                target, dtype
            )
            x.requires_grad = True
            x_mlu.requires_grad = True
            if weight_flag:
                weight_cpu = weight
                weight_mlu = self.to_mlu_dtype(weight, dtype)
            else:
                weight_cpu = None
                weight_mlu = None
            output = F.binary_cross_entropy(
                x, target, reduction=reduct, weight=weight_cpu
            )
            grad_cpu = torch.ones(output.shape)
            grad_mlu = self.to_mlu_dtype(grad_cpu, dtype)
            output.backward(grad_cpu)
            grad_input = copy.deepcopy(x.grad)
            x.grad.zero_()
            output_mlu = F.binary_cross_entropy(
                x_mlu, target_mlu, reduction=reduct, weight=weight_mlu
            )
            self.assertTrue(output_mlu.dtype == dtype)
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

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_bce_bfloat16(self):
        torch.manual_seed(1234)
        dtype = torch.bfloat16
        x = torch.rand((2, 3, 4, 5, 8), dtype=dtype).to(torch.float)
        target = torch.rand((2, 3, 4, 5, 8), dtype=dtype).to(torch.float)
        x_mlu, target_mlu = self.to_mlu_dtype(x, dtype), self.to_mlu_dtype(
            target, dtype
        )
        x.requires_grad = True
        x_mlu.requires_grad = True
        weight_cpu = None
        weight_mlu = None
        output = F.binary_cross_entropy(x, target, reduction="sum", weight=weight_cpu)
        grad_cpu = torch.ones(output.shape)
        grad_mlu = self.to_mlu_dtype(grad_cpu, dtype)
        output.backward(grad_cpu)
        out_grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        output_mlu = F.binary_cross_entropy(
            x_mlu, target_mlu, reduction="sum", weight=weight_mlu
        )
        output_mlu.backward(grad_mlu)
        out_grad_mlu = copy.deepcopy(x_mlu.grad)
        self.assertTensorsEqual(output, output_mlu.cpu().float(), 0.003, use_MSE=True)
        self.assertTensorsEqual(
            out_grad_cpu, out_grad_mlu.cpu().float(), 0.003, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("70GB")
    def test_bce_large_half(self):
        shape = (4, 1025, 1024, 1024)
        reduct = "mean"
        type_err = (torch.half, 3e-3)
        x = torch.rand(shape, dtype=torch.float)
        target = torch.rand(shape, dtype=torch.float)
        loss = nn.BCELoss(weight=None, reduction=reduct)
        loss_mlu = nn.BCELoss(weight=None, reduction=reduct)
        out_cpu = loss(x, target)
        out_mlu = loss_mlu(x.to("mlu"), target.to("mlu"))
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("69GB")
    def test_bce_bp_large(self):
        shape = (4, 1025, 1024, 1024)
        reduct = "mean"
        type_err = (torch.float, 3e-3)
        x = torch.rand(shape, dtype=torch.float, requires_grad=True).to(type_err[0])
        target = torch.rand(shape, dtype=torch.float).to(type_err[0])
        out_cpu = F.binary_cross_entropy(x, target, reduction=reduct)
        out_cpu.backward()
        grad_cpu = copy.deepcopy(x.grad)
        x.grad.zero_()
        out_mlu = F.binary_cross_entropy(
            x.to("mlu"), target.to("mlu"), reduction=reduct
        )
        out_mlu.backward()
        grad_mlu = copy.deepcopy(x.grad)
        x.grad.zero_()
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), type_err[1], use_MSE=True
        )
        self.assertTensorsEqual(
            grad_cpu.float(), grad_mlu.cpu().float(), type_err[1], use_MSE=True
        )


if __name__ == "__main__":
    run_tests()
