# pylint: disable=W0511
"""
# 编译安装torchaudio
## 安装条件
- pytorch2.1
- torch_mlu2.1
## 安装编译 audio
```bash
# audio 支持版本 2.1.1
git clone -b v2.1.1 https://github.com/pytorch/audio.git \
&& pushd audio && python setup.py install \
&& popd
```

"""
from __future__ import print_function

import unittest
import logging
import copy
from itertools import product
import sys
import os
import torch
import torch_mlu
from torchaudio import transforms

import itertools

torch.manual_seed(0)

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
# pylint: disable=C0413,C0411
from common_utils import testinfo, TestCase

logging.basicConfig(level=logging.DEBUG)  # NOSONAR


class TestRNNTLossOp(TestCase):
    """
    Test RNNTLoss
    """

    batch, max_seq_length, max_target_length, class_nums = 4, 4, 4, 6

    def rnnt_loss_input_with_params(
        self,
        batch,
        max_seq_length,
        max_target_length,
        class_nums,
        dtype=torch.float,
        device="cpu",
    ):
        logits_shape = [batch, max_seq_length, max_target_length + 1, class_nums]
        targets_shape = [batch, max_target_length]
        logit_lengths_shape = [batch]
        logits = torch.rand(
            logits_shape, dtype=dtype, device=device, requires_grad=True
        )
        targets = (
            torch.randint(2, size=targets_shape, dtype=torch.int32, device=device) + 1
        )
        logit_lengths = torch.randint(
            1,
            max_seq_length + 1,
            size=logit_lengths_shape,
            dtype=torch.int32,
            device=device,
        )
        target_lengths = torch.randint(
            1,
            max_target_length + 1,
            size=logit_lengths_shape,
            dtype=torch.int32,
            device=device,
        )

        if logits.size()[1] != logit_lengths.max().item():
            max_index = logit_lengths.argmax()
            logit_lengths[max_index] = logits.size()[1]
            logit_lengths = logit_lengths.clamp_max(logits.size()[1])
        assert (
            logits.size()[1] == logit_lengths.max().item()
        ), "logits size[1] => {} != {}".format(
            logits.size()[1], logit_lengths.max().item()
        )
        if logits.size()[2] != target_lengths.max().item() + 1:
            max_index = target_lengths.argmax()
            target_lengths[max_index] = logits.size()[2] - 1
            target_lengths.clamp_max(logits.size()[2] - 1)
        assert (
            logits.size()[2] == target_lengths.max().item() + 1
        ), "logits size[2] => {} != {}".format(
            logits.size()[2], target_lengths.max().item() + 1
        )
        return logits, targets, logit_lengths, target_lengths

    def forward_and_compare(
        self,
        batch,
        max_seq_length,
        max_target_length,
        class_nums,
        transform,
        dtype=torch.float,
        cmp_device="mlu",
    ):
        (
            logits,
            targets,
            logit_lengths,
            target_lengths,
        ) = self.rnnt_loss_input_with_params(
            batch, max_seq_length, max_target_length, class_nums, dtype=dtype
        )
        forward = transform(logits, targets, logit_lengths, target_lengths)
        forward_mlu = None
        if cmp_device == "mlu":
            forward_mlu = transform(
                logits.to(cmp_device),
                targets.to(cmp_device),
                logit_lengths.to(cmp_device),
                target_lengths.to(cmp_device),
            )
        return forward, forward_mlu

    def backward_and_compare(
        self,
        batch,
        max_seq_length,
        max_target_length,
        class_nums,
        transform=transforms.RNNTLoss(blank=0),
        dtype=torch.float,
        cmp_device="mlu",
    ):
        (
            logits,
            targets,
            logit_lengths,
            target_lengths,
        ) = self.rnnt_loss_input_with_params(
            batch, max_seq_length, max_target_length, class_nums, dtype=dtype
        )
        forward = transform(logits, targets, logit_lengths, target_lengths)
        forward.backward()
        grad_cpu = copy.deepcopy(logits.grad)
        logits.grad.zero_()
        if cmp_device == "mlu":
            logits_ = logits.mlu()
            forward_mlu = transform(
                logits_,
                self.to_mlu(targets),
                self.to_mlu(logit_lengths),
                self.to_mlu(target_lengths),
            )
            forward_mlu.backward()
            grad_mlu = copy.deepcopy(logits.grad)
        return grad_cpu, grad_mlu

    @testinfo()
    def test_backward_with_shape(self):
        params = [
            {
                "batch": 12,
                "max_seq_length": 10,
                "max_target_length": 15,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 12,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 16,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 18,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 20,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 16,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 18,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 20,
                "class_nums": 20,
            },
        ]
        for param in params:
            for dtype, blank, reduction, clamp in itertools.product(
                [torch.float, torch.half],
                range(5),
                ["mean", "sum"],
                map(lambda x: float(x), range(-5, 5, 1)),
            ):
                param["dtype"] = dtype
                param["transform"] = transforms.RNNTLoss(
                    blank=blank, reduction=reduction, clamp=clamp
                )
                cpu_forward, mlu_forward = self.backward_and_compare(**param)
                self.assertTensorsEqual(
                    cpu_forward.float(), mlu_forward.cpu().float(), 0.003, use_MSE=True
                )

    # @unittest.skip("not test")
    @testinfo()
    def test_forward_with_shape(self):
        params = [
            {
                "batch": 12,
                "max_seq_length": 10,
                "max_target_length": 15,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 12,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 16,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 18,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 20,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 10,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 16,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 18,
                "class_nums": 20,
            },
            {
                "batch": 3,
                "max_seq_length": 10,
                "max_target_length": 20,
                "class_nums": 20,
            },
        ]
        for param in params:
            for dtype, blank, reduction, clamp in itertools.product(
                [torch.float, torch.half],
                range(5),
                ["mean", "sum", "none"],
                list(map(lambda x: float(x), range(-5, 5, 1))),
            ):
                param["dtype"] = dtype
                param["transform"] = transforms.RNNTLoss(
                    blank=blank, reduction=reduction, clamp=clamp
                )
                cpu_forward, mlu_forward = self.forward_and_compare(**param)
                self.assertTensorsEqual(
                    cpu_forward.float(), mlu_forward.cpu().float(), 0.003, use_MSE=True
                )

    @testinfo()
    def test_rnnt_loss_forward_exception(self):
        (
            logits,
            targets,
            logit_lengths,
            target_lengths,
        ) = self.rnnt_loss_input_with_params(
            self.batch, self.max_seq_length, self.max_target_length, self.class_nums
        )
        logit_lengths_illegal = logit_lengths.clamp_max(logits.size()[1] - 1)
        target_lengths_illegal = target_lengths.clamp_max(logits.size()[2] - 2)
        targets_illegal = torch.ones_like(targets.transpose(0, 1))
        input_length_msg = r"^input length mismatch$"
        output_length_msg = r"^output length mismatch$"

        transform = transforms.RNNTLoss(blank=0)
        with self.assertRaisesRegex(RuntimeError, input_length_msg):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                logit_lengths_illegal.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(RuntimeError, output_length_msg):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths_illegal.to("mlu"),
            )

        with self.assertRaisesRegex(RuntimeError, r"^targets must be contiguous$"):
            transform(
                logits.to("mlu"),
                targets.to("mlu").transpose(0, 1),
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(RuntimeError, r"^targets must be contiguous$"):
            transform(
                logits.to("mlu"),
                targets_illegal.to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"Expected all tensors to be on the same device"
        ):
            transform(
                logits.to("mlu"),
                targets,
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"Expected all tensors to be on the same device"
        ):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                logit_lengths,
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"^logits must be float32 or float16 \(half\) type$"
        ):
            transform(
                logits.type(torch.int).to("mlu"),
                targets.to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(RuntimeError, r"^targets must be int32 type$"):
            transform(
                logits.to("mlu"),
                targets.type(torch.float).to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"^logit_lengths must be int32 type$"
        ):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                logit_lengths.type(torch.float).to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"^logits must be 4-D \(batch, time, target, class\)$"
        ):
            transform(
                logits.reshape(-1).to("mlu"),
                targets.to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"^targets must be 2-D \(batch, max target length\)$"
        ):
            transform(
                logits.to("mlu"),
                targets.reshape(-1).to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(RuntimeError, r"^logit_lengths must be 1-D$"):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                logit_lengths.reshape(2, 2).to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(RuntimeError, r"^target_lengths must be 1-D$"):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths.reshape(2, 2).to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"^batch dimension mismatch between logits and logit_lengths$"
        ):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                torch.ones(logits.size(0) + 1, dtype=torch.int).to("mlu"),
                target_lengths.to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"^batch dimension mismatch between logits and target_lengths$",
        ):
            transform(
                logits.to("mlu"),
                targets.to("mlu"),
                logit_lengths.to("mlu"),
                torch.ones(logits.size(0) + 1, dtype=torch.int).to("mlu"),
            )

        with self.assertRaisesRegex(
            RuntimeError, r"^batch dimension mismatch between logits and targets$"
        ):
            transform(
                logits.to("mlu"),
                torch.ones(
                    [self.batch + 1, self.max_target_length], dtype=torch.int
                ).to("mlu"),
                logit_lengths.to("mlu"),
                target_lengths.to("mlu"),
            )


if __name__ == "__main__":
    unittest.main()
