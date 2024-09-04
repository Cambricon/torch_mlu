from __future__ import print_function

import sys
import os
import copy
import random
import itertools
import unittest
import logging

# from warpctc_pytorch import CTCLoss  # baidu's warpctc
import torch
import torch.nn as nn
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestCTCLossOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_warp_ctc_loss(self):
        T = 50
        C = 20  # Number of classes (including blank)
        N = 16
        S = 30  # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        probs = torch.randn(T, N, C).requires_grad_()
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        target = torch.randint(
            low=1, high=C, size=(sum(target_lengths),), dtype=torch.long
        )
        # NOTE:
        # 1. outshape of baidu's warpctc is [1], but outshape of pytorch is []
        # 2. input of baidu's warpctc doesn't go through log_softmax.
        # ctc_loss = CTCLoss()  # baidu's warpctc
        ctc_loss = nn.CTCLoss(reduction="sum", zero_infinity=True)  # pytorch's ctc
        out_cpu = ctc_loss(probs.log_softmax(2), target, input_lengths, target_lengths)
        out_cpu.backward()
        grad_cpu = copy.deepcopy(probs.grad)
        probs.grad.zero_()
        out_mlu = ctc_loss(
            probs.log_softmax(2).to("mlu"),
            target.to("mlu"),
            input_lengths,
            target_lengths,
        )
        out_mlu.backward()
        grad_mlu = copy.deepcopy(probs.grad)
        probs.grad.zero_()
        self.assertTensorsEqual(
            out_cpu.float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_MSE=True)

        out_mlu = torch.ops.torch_mlu.warp_ctc_loss(
            probs.to("mlu"),
            target.to("mlu"),
            input_lengths.to("mlu"),
            target_lengths.to("mlu"),
            0,
            1,
            True,
            0,
        )
        out_mlu.backward()
        grad_mlu = copy.deepcopy(probs.grad)
        probs.grad.zero_()
        self.assertTensorsEqual(
            out_cpu.unsqueeze(0).float(), out_mlu.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(grad_cpu, grad_mlu, 0.003, use_MSE=True)

    def compare_ctc_loss(
        self,
        log_probs,
        target,
        input_lengths,
        target_lengths,
        reduction="mean",
        blank=0,
        zero_infinity=False,
        use_MSE=True,
        diff_f=0.003,
        diff_b=0.003,
    ):
        ctc_loss = nn.CTCLoss(
            reduction=reduction, blank=blank, zero_infinity=zero_infinity
        )
        out_cpu = ctc_loss(log_probs, target, input_lengths, target_lengths)
        out_cpu.sum().backward()
        grad_cpu = copy.deepcopy(log_probs.grad)
        log_probs.grad.zero_()
        out_mlu = ctc_loss(
            log_probs.to("mlu"), target.to("mlu"), input_lengths, target_lengths
        )
        out_mlu.sum().backward()
        grad_mlu = copy.deepcopy(log_probs.grad)
        log_probs.grad.zero_()
        if use_MSE:
            self.assertTensorsEqual(
                out_cpu.float(), out_mlu.cpu().float(), diff_f, use_MSE=True
            )
            self.assertTensorsEqual(grad_cpu, grad_mlu, diff_b, use_MSE=True)
        else:
            self.assertEqual(out_cpu.float(), out_mlu.cpu().float())
            self.assertEqual(grad_cpu, grad_mlu)

    # @unittest.skip("not test")
    @testinfo()
    def test_ctc_loss(self):
        T = 50
        C = 20  # Number of classes (including blank)
        N = 16
        S = 30  # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        log_probs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        input_lengths_list = [
            torch.full(size=(N,), fill_value=T, dtype=torch.long),
            torch.full(size=(N,), fill_value=T, dtype=torch.long).numpy().tolist(),
        ]
        target_lengths_list = [
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long),
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
            .numpy()
            .tolist(),
        ]
        reduction_list = ["none", "mean", "sum"]
        zero_infinity_list = [True, False]
        for reduction, zero_infinity in itertools.product(
            reduction_list, zero_infinity_list
        ):
            for input_lengths, target_lengths in zip(
                input_lengths_list, target_lengths_list
            ):
                # target shape = (N, S)
                target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
                self.compare_ctc_loss(
                    log_probs,
                    target,
                    input_lengths,
                    target_lengths,
                    reduction=reduction,
                    zero_infinity=zero_infinity,
                )
                # test blank
                blank = random.randint(int((C - 1) / 2), C - 1)
                # target should not include blank
                target = torch.randint(low=0, high=blank, size=(N, S), dtype=torch.long)
                self.compare_ctc_loss(
                    log_probs,
                    target,
                    input_lengths,
                    target_lengths,
                    reduction=reduction,
                    blank=blank,
                    zero_infinity=zero_infinity,
                )

                # target shape = sum(target_lengths)
                target = torch.randint(
                    low=1, high=C, size=(sum(target_lengths),), dtype=torch.long
                )
                self.compare_ctc_loss(
                    log_probs,
                    target,
                    input_lengths,
                    target_lengths,
                    reduction=reduction,
                    zero_infinity=zero_infinity,
                )
                # test blank
                blank = random.randint(int((C - 1) / 2), C - 1)
                # target should not include blank
                target = torch.randint(
                    low=0, high=blank, size=(sum(target_lengths),), dtype=torch.long
                )
                self.compare_ctc_loss(
                    log_probs,
                    target,
                    input_lengths,
                    target_lengths,
                    reduction=reduction,
                    blank=blank,
                    zero_infinity=zero_infinity,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_ctc_loss_zero_infinity(self):
        T = 10  # when seqLength < targetNum, loss is 0.0 and probs.grad is 0.0
        C = 20  # Number of classes (including blank)
        N = 16
        S = 30  # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        log_probs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        # target shape = (N, S)
        target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        input_lengths_list = [
            torch.full(size=(N,), fill_value=T, dtype=torch.long),
            torch.full(size=(N,), fill_value=T, dtype=torch.long).numpy().tolist(),
        ]
        target_lengths_list = [
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long),
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
            .numpy()
            .tolist(),
        ]
        reduction_list = ["none", "mean", "sum"]
        zero_infinity_list = [True, False]
        for reduction, zero_infinity in itertools.product(
            reduction_list, zero_infinity_list
        ):
            for input_lengths, target_lengths in zip(
                input_lengths_list, target_lengths_list
            ):
                self.compare_ctc_loss(
                    log_probs,
                    target,
                    input_lengths,
                    target_lengths,
                    reduction=reduction,
                    zero_infinity=zero_infinity,
                    use_MSE=False,
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_ctc_loss_exceptions(self):
        device = "mlu"
        target_lengths = [30, 25, 20]
        input_lengths = [50, 50, 51]
        targets = torch.randint(1, 15, (3, 30), dtype=torch.long, device=device)
        log_probs = torch.randn(
            50, 3, 15, dtype=torch.float, device=device
        ).log_softmax(2)
        ref_msg = r"Expected input_lengths to have value at most 50, but got value 51"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths
            )
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (3, 29), dtype=torch.long, device=device)
        ref_msg = (
            r"Expected tensor to have size at least 30 at dimension 1, but got size 29"
        )
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.nn.functional.ctc_loss(
                log_probs, targets, input_lengths, target_lengths
            )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("15GB", device="mlu")
    def test_ctc_loss_custom_case(self):
        T = 1109
        N = 48
        C = 13725  # Number of classes (including blank)
        S = 35  # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        log_probs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        input_lengths_list = [
            torch.full(size=(N,), fill_value=T, dtype=torch.long),
            torch.full(size=(N,), fill_value=T, dtype=torch.long).numpy().tolist(),
        ]
        target_lengths_list = [
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long),
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
            .numpy()
            .tolist(),
        ]
        reduction_list = ["none", "mean", "sum"]
        zero_infinity_list = [True, False]
        for reduction, zero_infinity in itertools.product(
            reduction_list, zero_infinity_list
        ):
            for input_lengths, target_lengths in zip(
                input_lengths_list, target_lengths_list
            ):
                # target shape = sum(target_lengths)
                target = torch.randint(
                    low=1, high=C, size=(sum(target_lengths),), dtype=torch.long
                )
                self.compare_ctc_loss(
                    log_probs,
                    target,
                    input_lengths,
                    target_lengths,
                    reduction=reduction,
                    zero_infinity=zero_infinity,
                    diff_b=0.05,
                )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("28GB", device="mlu")
    def test_ctc_loss_large(self):
        T = 4096
        N = 48
        C = 13725  # Number of classes (including blank)
        S = 80  # Target sequence length of longest target in batch (padding length)
        S_min = 10  # Minimum target length, for demonstration purposes

        log_probs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        input_lengths_list = [
            torch.full(size=(N,), fill_value=T, dtype=torch.long),
            torch.full(size=(N,), fill_value=T, dtype=torch.long).numpy().tolist(),
        ]
        target_lengths_list = [
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long),
            torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
            .numpy()
            .tolist(),
        ]
        reduction_list = ["none", "mean", "sum"]
        zero_infinity_list = [True, False]
        for reduction, zero_infinity in itertools.product(
            reduction_list, zero_infinity_list
        ):
            for input_lengths, target_lengths in zip(
                input_lengths_list, target_lengths_list
            ):
                # target shape = sum(target_lengths)
                target = torch.randint(
                    low=1, high=C, size=(sum(target_lengths),), dtype=torch.long
                )
                self.compare_ctc_loss(
                    log_probs,
                    target,
                    input_lengths,
                    target_lengths,
                    reduction=reduction,
                    zero_infinity=zero_infinity,
                    diff_b=0.05,
                )


if __name__ == "__main__":
    run_tests()
