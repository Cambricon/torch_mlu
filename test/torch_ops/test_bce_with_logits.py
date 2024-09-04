from __future__ import print_function

import sys
import os
import copy
import unittest
import logging
import torch
import torch.nn.functional as F
from itertools import product

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_BFLOAT16,
    skipBFloat16IfNotSupport,
    TEST_LARGETENSOR,
    largeTensorTest,
)  # pylint: disable=C0411, C0413

logging.basicConfig(level=logging.DEBUG)


class TestBceWithLogitsOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_bce_with_logits(self):
        def run_test(x, target, weight, pos_weight):
            x_mlu, target_mlu = x.to("mlu"), target.to("mlu")
            if x.dtype == torch.half:
                x = x.float()
                target = target.float()
            reduct_lst = ["none", "mean", "sum"]
            for reduct in reduct_lst:
                for weight_flag in [True, False]:
                    if weight_flag:
                        weight_ = (
                            weight.float() if weight.dtype == torch.half else weight
                        )
                        weight_mlu = weight.to("mlu")
                        pos_weight_ = (
                            pos_weight.float()
                            if pos_weight.dtype == torch.half
                            else pos_weight
                        )
                        pos_weight_mlu = pos_weight.to("mlu")
                    else:
                        weight_ = None
                        weight_mlu = None
                        pos_weight_ = None
                        pos_weight_mlu = None
                    out_cpu = F.binary_cross_entropy_with_logits(
                        x,
                        target,
                        reduction=reduct,
                        weight=weight_,
                        pos_weight=pos_weight_,
                    )
                    out_mlu = F.binary_cross_entropy_with_logits(
                        x_mlu,
                        target_mlu,
                        reduction=reduct,
                        weight=weight_mlu,
                        pos_weight=pos_weight_mlu,
                    )
                    self.assertTensorsEqual(
                        out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
                    )

        shape_list = [(527, 80), (32, 3, 26, 26)]
        dtype_list = [torch.half, torch.float, torch.double]
        for shape, dtype in product(shape_list, dtype_list):
            x = torch.rand(shape, dtype=dtype)
            target = torch.rand(shape, dtype=dtype)
            weight = torch.rand(shape, dtype=dtype)
            pos_weight = torch.rand(shape, dtype=dtype)
            # test contiguous
            run_test(x, target, weight, pos_weight)
            if x.dim() == 4:
                # test channels_last
                run_test(
                    x.to(memory_format=torch.channels_last),
                    target.to(memory_format=torch.channels_last),
                    weight.to(memory_format=torch.channels_last),
                    pos_weight.to(memory_format=torch.channels_last),
                )

                # test no dense
                run_test(
                    x[:, :, :, :2],
                    target[:, :, :, :2],
                    weight[:, :, :, :2],
                    pos_weight[:, :, :, :2],
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_with_logits_bp(self):
        def run_test(x, target, weight, grad_in, pos_weight):
            x.requires_grad = True
            reduct_lst = ["none", "mean", "sum"]
            for reduct in reduct_lst:
                grad_in_mlu = grad_in.to("mlu")
                for weight_flag in [True, False]:
                    if weight_flag:
                        weight_ = weight
                        weight_mlu = weight.to("mlu")
                        pos_weight_ = pos_weight
                        pos_weight_mlu = pos_weight.to("mlu")
                    else:
                        weight_ = None
                        weight_mlu = None
                        pos_weight_ = None
                        pos_weight_mlu = None
                    out_cpu = F.binary_cross_entropy_with_logits(
                        x,
                        target,
                        reduction=reduct,
                        weight=weight_,
                        pos_weight=pos_weight_,
                    )
                    if reduct == "none":
                        out_cpu.backward(grad_in)
                    else:
                        out_cpu.backward()
                    grad_cpu = copy.deepcopy(x.grad)
                    x.grad.zero_()
                    out_mlu = F.binary_cross_entropy_with_logits(
                        x.to("mlu"),
                        target.to("mlu"),
                        reduction=reduct,
                        weight=weight_mlu,
                        pos_weight=pos_weight_mlu,
                    )
                    if reduct == "none":
                        out_mlu.backward(grad_in_mlu)
                    else:
                        out_mlu.backward()
                    grad_mlu = copy.deepcopy(x.grad)
                    x.grad.zero_()
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    self.assertTensorsEqual(
                        grad_cpu, grad_mlu.cpu(), 0.003, use_MSE=True
                    )

        shape_list = [(527, 80, 2, 3), (32, 3, 14, 26)]
        for shape in shape_list:
            x = torch.rand(shape, dtype=torch.float, requires_grad=True)
            target = torch.rand(shape, dtype=torch.float)
            weight = torch.rand(shape, dtype=torch.float)
            pos_weight = torch.rand(shape, dtype=torch.float)
            grad_in = torch.rand(shape, dtype=torch.float)
            # test contiguous
            run_test(x, target, weight, grad_in, pos_weight)

            # test no dense
            run_test(
                x[:, :, :, :2].detach(),
                target[:, :, :, :2],
                weight[:, :, :, :2],
                grad_in[:, :, :, :2],
                pos_weight[:, :, :, :2],
            )

            # tes channels last
            run_test(
                x.to(memory_format=torch.channels_last).detach(),
                target.to(memory_format=torch.channels_last),
                weight.to(memory_format=torch.channels_last),
                grad_in.to(memory_format=torch.channels_last),
                pos_weight.to(memory_format=torch.channels_last),
            )

    # @unittest.skip("not test")
    @testinfo()
    def test_bce_with_logits_exception(self):
        a = torch.randn(2, 3).to("mlu")
        b = torch.randn(2, 3).int().to("mlu")
        ref_msg = "Expected same dtype of self and target, "
        ref_msg += (
            "but got dtype float for argument 'self' and int for argument 'target'"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            out = F.binary_cross_entropy_with_logits(a, b)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_bce_with_logits_bfloat16(self):
        shape = (527, 80)
        data_type = torch.bfloat16
        x = torch.rand(shape, dtype=data_type)
        target = torch.rand(shape, dtype=data_type)
        weight = torch.rand(shape, dtype=data_type)
        pos_weight = torch.rand(shape, dtype=data_type)

        # compute on cpu
        out_cpu = F.binary_cross_entropy_with_logits(
            x.float(),
            target.float(),
            reduction="none",
            weight=weight.float(),
            pos_weight=pos_weight.float(),
        )

        # compute on mlu
        out_mlu = F.binary_cross_entropy_with_logits(
            x.mlu(),
            target.mlu(),
            reduction="none",
            weight=weight.mlu(),
            pos_weight=pos_weight.mlu(),
        )

        # check the result
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("26GB")
    @unittest.skip("not test")
    def test_bce_with_logits_large(self):
        shape = (4 * 1025, 1024 * 1024)
        data_type = torch.float
        x = torch.rand(shape, dtype=data_type)
        target = torch.rand(shape, dtype=data_type)

        # compute on cpu
        out_cpu = F.binary_cross_entropy_with_logits(x, target, reduction="none")

        # compute on mlu
        out_mlu = F.binary_cross_entropy_with_logits(
            x.mlu(), target.mlu(), reduction="none"
        )
        # check the result
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
