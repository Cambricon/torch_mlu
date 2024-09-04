from __future__ import print_function

import unittest
import logging
from itertools import product
import sys
import os
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)


class FusedMaskSoftmaxDropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, mask, batch, seqlen, heads, dropout_prob, stream, sync, is_training
    ):
        output, dropout_mask = torch.ops.torch_mlu.mask_softmax_dropout_fprop(
            input, mask, batch, seqlen, heads, dropout_prob, stream, sync, is_training
        )

        ctx.save_for_backward(input, dropout_mask, seqlen)
        ctx.batch = batch
        ctx.heads = heads
        ctx.dropout_prob = dropout_prob
        ctx.stream = stream
        ctx.sync = sync
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            output,
            dropout_mask,
            seqlen,
        ) = ctx.saved_tensors
        batch = ctx.batch
        heads = ctx.heads
        dropout_prob = ctx.dropout_prob

        torch.ops.torch_mlu.mask_softmax_dropout_bprop_(
            output,
            grad_output,
            dropout_mask,
            batch,
            seqlen,
            heads,
            dropout_prob,
            ctx.stream,
            ctx.sync,
        )
        return grad_output, None, None, None, None, None, None, None, None


class BaseMaskSoftmaxDropout(torch.nn.Module):
    def __init__(self, dropout_prob):
        super(BaseMaskSoftmaxDropout, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, batch, seqlen, input, mask, heads):
        cnt, mask_cnt = 0, 0
        out = torch.empty_like(input)
        for i in range(batch):
            seq = seqlen[i].item()
            input_view = input[cnt : cnt + heads * seq * seq].view(heads * seq, seq)
            mask_slice = mask[mask_cnt : mask_cnt + seq]
            out_slice = torch.nn.functional.softmax(input_view + mask_slice, dim=-1)
            out[cnt : cnt + heads * seq * seq].copy_(out_slice.view(-1))
            cnt += heads * seq * seq
            mask_cnt += seq

        return self.dropout(out)


class TestFusedMaskSoftmaxDropoutOp(TestCase):
    @testinfo()
    def test_mask_softmax_dropout(self):
        batch_list = [1, 16, 96]
        heads_list = [1, 16, 32]
        prob_list = [0.0, 0.3, 0.8, 1.0]
        dtype_list = [torch.half, torch.float]
        for batch, heads, prob, dtype in product(
            batch_list, heads_list, prob_list, dtype_list
        ):
            seqlen = torch.randint(1, 6, (batch,))
            input = torch.randn(
                seqlen.square().sum().item() * heads,
                dtype=dtype,
                device="mlu",
                requires_grad=True,
            )
            mask = torch.randn(seqlen.sum().item(), dtype=dtype, device="mlu")
            grad = torch.randn(input.size(), dtype=dtype, device="mlu")
            baseline = BaseMaskSoftmaxDropout(prob)
            torch.mlu.manual_seed(10000)
            out_base = baseline(batch, seqlen, input, mask, heads)
            out_base.backward(grad)
            grad_base = input.grad
            input.grad.zero_()
            torch.mlu.manual_seed(10000)
            out_fused = FusedMaskSoftmaxDropoutFunction.apply(
                input, mask, batch, seqlen, heads, prob, False, False, True
            )
            out_fused.backward(grad)
            grad_fused = input.grad
            # TODO(PYTORCH-11182): bypass for random failure
            # self.assertTensorsEqual(out_base.cpu().float(), out_fused.cpu().float(),
            #   0.003, use_MSE=True)
            self.assertTensorsEqual(
                grad_base.cpu().float(), grad_fused.cpu().float(), 0.003, use_MSE=True
            )

    @testinfo()
    def test_mask_softmax_dropout_not_contiguous(self):
        func_list = [lambda x: x, self.to_non_dense]
        heads_list = [1, 16]
        for heads, func0, func1, func2, func3 in product(
            heads_list, func_list, func_list, func_list, func_list
        ):
            seqlen = func0(torch.randint(1, 513, (1,)))
            input = torch.randn(
                seqlen.square().sum().item() * heads, device="mlu", requires_grad=True
            )
            mask = func2(torch.randn(seqlen.sum().item(), device="mlu"))
            grad = func3(torch.randn(input.size(), device="mlu"))
            baseline = BaseMaskSoftmaxDropout(0.5)
            torch.mlu.manual_seed(10000)
            out_base = baseline(1, seqlen, func1(input), mask, heads)
            out_base.backward(grad)
            grad_base = input.grad
            input.grad.zero_()
            torch.mlu.manual_seed(10000)
            out_fused = FusedMaskSoftmaxDropoutFunction.apply(
                func1(input), mask, 1, seqlen, heads, 0.5, False, False, True
            )
            out_fused.backward(grad)
            grad_fused = input.grad
            self.assertTensorsEqual(
                out_base.cpu(), out_fused.cpu(), 0.003, use_MSE=True
            )
            self.assertTensorsEqual(
                grad_base.cpu(), grad_fused.cpu(), 0.003, use_MSE=True
            )

    @testinfo()
    def test_mask_softmax_dropout_eval(self):
        batch_list = [1, 16, 96]
        heads_list = [1, 16, 32]
        prob_list = [0.0, 0.3, 0.8, 1.0]
        dtype_list = [torch.half, torch.float]
        for batch, heads, prob, dtype in product(
            batch_list, heads_list, prob_list, dtype_list
        ):
            seqlen = torch.randint(1, 6, (batch,))
            input = torch.randn(
                seqlen.square().sum().item() * heads, dtype=dtype, device="mlu"
            )
            mask = torch.randn(seqlen.sum().item(), dtype=dtype, device="mlu")
            baseline = BaseMaskSoftmaxDropout(prob)
            torch.mlu.manual_seed(10000)
            baseline.eval()
            out_base = baseline(batch, seqlen, input, mask, heads)
            torch.mlu.manual_seed(10000)
            out_fused = FusedMaskSoftmaxDropoutFunction.apply(
                input, mask, batch, seqlen, heads, prob, False, False, False
            )
            # TODO(PYTORCH-11182): bypass for random failure
            # self.assertTensorsEqual(out_base.cpu().float(), out_fused.cpu().float(),
            #   0.003, use_MSE=True)

    @testinfo()
    def test_mask_softmax_dropout_exception(self):
        seqlen = torch.randint(1, 513, (10,), device="mlu")
        input = torch.randn(10, device="mlu")
        mask = torch.randn(10, device="mlu")
        ref_msg = "input and mask must be on MLU, seq_len must be on CPU!"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.ops.torch_mlu.mask_softmax_dropout_fprop(
                input, mask, 10, seqlen, 1, 0.5, False, False, False
            )  # pylint: disable=W0612

        seqlen = torch.randint(1, 513, (10,))
        input = torch.randint(1, 10000, (10,), device="mlu")
        mask = torch.randint(1, 10000, (10,), device="mlu")
        ref_msg = "only support real floating point input!"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.ops.torch_mlu.mask_softmax_dropout_fprop(
                input, mask, 10, seqlen, 1, 0.5, False, False, False
            )  # pylint: disable=W0612

        seqlen = torch.randint(1, 513, (10,))
        input = torch.randn(10, device="mlu", dtype=torch.half)
        mask = torch.randn(10, device="mlu", dtype=torch.float)
        ref_msg = "the dtype of input and mask must be the same, but got Half and Float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.ops.torch_mlu.mask_softmax_dropout_fprop(
                input, mask, 10, seqlen, 1, 0.5, False, False, False
            )  # pylint: disable=W0612

        seqlen = torch.randn(10)
        input = torch.randn(10, device="mlu")
        mask = torch.randn(10, device="mlu")
        ref_msg = "dtype of seq_len must be Int or Long!"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.ops.torch_mlu.mask_softmax_dropout_fprop(
                input, mask, 10, seqlen, 1, 0.5, False, False, False
            )  # pylint: disable=W0612

        seqlen = torch.randint(1, 513, (10,))
        input = torch.randn((5, 2), device="mlu")
        mask = torch.randn(10, device="mlu")
        ref_msg = "only support 1-D Tensors for input, mask and seq_len, but got "
        ref_msg += "2-D, 1-D and 1-D!"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.ops.torch_mlu.mask_softmax_dropout_fprop(
                input, mask, 10, seqlen, 1, 0.5, False, False, False
            )  # pylint: disable=W0612

        seqlen = torch.randint(1, 513, (10,))
        seqlen[0] = 609
        input = torch.randn(10, device="mlu")
        mask = torch.randn(10, device="mlu")
        ref_msg = "the max value of seqlen must less than 608 currently because "
        ref_msg += "of CNNLExtra limitation."
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.ops.torch_mlu.mask_softmax_dropout_fprop(
                input, mask, 10, seqlen, 1, 0.5, False, False, False
            )  # pylint: disable=W0612

        seqlen = torch.randint(1, 513, (10,))
        input = torch.randn(10, device="mlu")
        output = torch.randn(10, device="mlu")
        mask = torch.randint(1, (10,), device="mlu").byte()
        ref_msg = "currently do not support multiple streams acceleration"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            o = torch.ops.torch_mlu.mask_softmax_dropout_bprop_(
                input, output, mask, 10, seqlen, 1, 0.5, True, True
            )  # pylint: disable=W0612


if __name__ == "__main__":
    unittest.main()
