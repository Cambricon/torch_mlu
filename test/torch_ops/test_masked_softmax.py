from __future__ import print_function

import os
import sys
import torch
import unittest
from itertools import product

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


class TestMaskedSoftmax(TestCase):
    def _slow_masked_softmax(self, input, mask, dim):
        exp = torch.exp(input)
        exp = exp * mask
        s = torch.sum(exp, dim=dim, keepdim=True).expand(exp.size())
        return exp / s

    @testinfo()
    def test_masked_softmax_mask_types(self):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]

        for B, num_heads, L in sizes:
            # mask_type == 0 => attention mask of shape LxL
            src_mask_orig = torch.randint(0, 2, (L, L)).bool()
            src_mask = (
                src_mask_orig.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()
            )

            # mask_type == 1 => padding mask of shape BxL
            src_key_padding_mask_orig = torch.randint(0, 2, (B, L)).bool()
            src_key_padding_mask = (
                src_key_padding_mask_orig.reshape(B, 1, 1, L)
                .expand(B, num_heads, L, L)
                .bool()
            )

            # mask_type == 2 =>  shape BxHxLxL
            generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()
            masks = [
                (src_mask_orig, src_mask, 0),
                (src_key_padding_mask_orig, src_key_padding_mask, 1),
                (generic_mask, generic_mask, 2),
            ]
            for dim in [-1, 3]:
                for mask_orig, mask, mask_type in masks:
                    input = torch.randn((B, num_heads, L, L))
                    input = input.mlu()
                    mask = mask.mlu()
                    mask_orig = mask_orig.mlu()
                    native_res = torch._masked_softmax(input, mask_orig, dim, mask_type)
                    mask = ~mask

                    pt_res = self._slow_masked_softmax(input, mask, dim)
                    pt_res = torch.nan_to_num(pt_res)

                    mask_not = mask.logical_not()

                    # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                    # Converts rows with all True's to False
                    mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                    self.assertEqual(
                        pt_res.masked_fill(mask_out, 0),
                        native_res.masked_fill(mask_out, 0),
                        exact_dtype=True,
                    )

    @testinfo()
    def test_masked_softmax_devices_parity(self):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for B, num_heads, L in sizes:
            # mask_type == 0 => attention mask of shape LxL
            src_mask = torch.randint(0, 2, (L, L)).bool()
            # mask_type == 1 => padding mask of shape BxL
            src_key_padding_mask = torch.randint(0, 2, (B, L)).bool()
            # mask_type == 2 => generic mask of shape BxHxLxL
            generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()
            masks = [(src_mask, 0), (src_key_padding_mask, 1), (generic_mask, 2)]
            input = torch.randn((B, num_heads, L, L))

            for dim in [-1, 3]:
                for mask, mask_type in masks:

                    def softmax_on_device(mask, input, device):
                        # Compute softmax on a given device
                        input_device = input.to(device)
                        mask_device = mask.to(device)
                        softmax_res = torch._masked_softmax(
                            input_device, mask_device, dim, mask_type
                        )
                        if mask_type == 0:
                            mask_expanded = (
                                mask_device.reshape(1, 1, L, L)
                                .expand(B, num_heads, L, L)
                                .bool()
                            )
                        elif mask_type == 1:
                            mask_expanded = (
                                mask_device.reshape(B, 1, 1, L)
                                .expand(B, num_heads, L, L)
                                .bool()
                            )
                        else:
                            mask_expanded = mask_device
                        # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                        # Fill rows with all True's with 0
                        mask_out = mask_expanded.all(dim, keepdim=True).expand(
                            mask_expanded.shape
                        )
                        softmax_res = softmax_res.masked_fill(mask_out, 0)
                        return softmax_res

                    cpu_res = softmax_on_device(mask, input, "cpu")
                    mlu_res = softmax_on_device(mask, input, "mlu")
                    self.assertEqual(cpu_res, mlu_res, exact_dtype=True)

    @testinfo()
    def test_masked_softmax_dtype(self):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for B, num_heads, L in sizes:
            src_mask = torch.randint(0, 2, (L, L)).bool()

            src_key_padding_mask = torch.randint(0, 2, (B, L)).bool()

            generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()

            masks = [(src_mask, 0), (src_key_padding_mask, 1), (generic_mask, 2)]

            dims = [-1, 3]
            dtypes = [torch.float, torch.half]
            for dim, dtype in product(dims, dtypes):
                input = torch.randn((B, num_heads, L, L), dtype=dtype)

                for mask, mask_type in masks:

                    def softmax_on_device(mask, input, device):
                        input_device = input.to(device)
                        mask_device = mask.to(device)
                        softmax_res = torch._masked_softmax(
                            input_device, mask_device, dim, mask_type
                        )
                        if mask_type == 0:
                            mask_expanded = (
                                mask_device.reshape(1, 1, L, L)
                                .expand(B, num_heads, L, L)
                                .bool()
                            )
                        elif mask_type == 1:
                            mask_expanded = (
                                mask_device.reshape(B, 1, 1, L)
                                .expand(B, num_heads, L, L)
                                .bool()
                            )
                        else:
                            mask_expanded = mask_device
                        mask_out = mask_expanded.all(dim, keepdim=True).expand(
                            mask_expanded.shape
                        )
                        softmax_res = softmax_res.masked_fill(mask_out, 0)
                        return softmax_res

                    cpu_res = softmax_on_device(mask, input.float(), "cpu")
                    mlu_res = softmax_on_device(mask, input, "mlu")
                    self.assertTensorsEqual(
                        cpu_res, mlu_res.cpu().float(), 0.003, use_MSE=True
                    )

    @testinfo()
    def test_masked_softmax_transformer_layout(self):
        B = 211
        num_heads = 16
        L = 42
        input = torch.randn((B, num_heads, L, L))
        dim = input.dim() - 1
        mask = torch.randint(0, 2, (B, L))
        mask_type = 1  # BxL => src_key_padding_mask
        input = input.mlu()
        mask = mask.mlu()
        mask = mask.bool()
        native_res = torch._masked_softmax(input, mask, dim, mask_type)
        mask = mask.reshape(B, 1, 1, L).expand(B, num_heads, L, L)
        mask = ~mask

        pt_res = self._slow_masked_softmax(input, mask, dim=dim)
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    @testinfo()
    def test_masked_softmax_TxT_layout(self):
        B = 211
        num_heads = 16
        L = 42
        input = torch.randn((B, num_heads, L, L))
        dim = input.dim() - 1
        mask = torch.randint(0, 2, (L, L))
        mask_type = 0  # LxL => src_mask
        input = input.mlu()
        mask = mask.mlu()
        mask = mask.bool()
        native_res = torch._masked_softmax(input, mask, dim, mask_type)
        mask = mask.expand(B, num_heads, L, L)
        mask = ~mask

        pt_res = self._slow_masked_softmax(input, mask, dim=dim)
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    @testinfo()
    def test_masked_softmax_not_dense(self):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]

        for B, num_heads, L in sizes:
            src_mask_orig = torch.randint(0, 2, (L, L)).bool()
            src_mask = (
                src_mask_orig.reshape(1, 1, L, L).expand(B, num_heads, L, L).bool()
            )

            src_key_padding_mask_orig = torch.randint(0, 2, (B, L)).bool()
            src_key_padding_mask = (
                src_key_padding_mask_orig.reshape(B, 1, 1, L)
                .expand(B, num_heads, L, L)
                .bool()
            )

            generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()
            masks = [
                (src_mask_orig, src_mask, 0),
                (src_key_padding_mask_orig, src_key_padding_mask, 1),
                (generic_mask, generic_mask, 2),
            ]

            for dim in [-1, 3]:
                for mask_orig, mask, mask_type in masks:
                    input = torch.randn((B, num_heads, L, L))
                    input = input.mlu()
                    mask = mask.mlu()
                    mask_orig = mask_orig.mlu()
                    if mask_type == 2:
                        mask_orig = mask_orig[:, :, :3, :3]
                    elif mask_type == 1:
                        mask_orig = mask_orig[:, :3]
                    else:
                        mask_orig = mask_orig[:3, :3]

                    native_res = torch._masked_softmax(
                        input[:, :, :3, :3], mask_orig, dim, mask_type
                    )

                    mask = ~mask
                    pt_res = self._slow_masked_softmax(
                        input[:, :, :3, :3], mask[:, :, :3, :3], dim
                    )
                    pt_res = torch.nan_to_num(pt_res)

                    mask_not = mask[:, :, :3, :3].logical_not()

                    mask_out = mask_not.all(dim, keepdim=True).expand(mask_not.shape)
                    self.assertEqual(
                        pt_res.masked_fill(mask_out, 0),
                        native_res.masked_fill(mask_out, 0),
                        exact_dtype=True,
                    )

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("48GB")
    def test_masked_softmax_large(self):
        B = 32
        num_heads = 16
        L = 2048
        input = torch.randn((B, num_heads, L, L))
        dim = input.dim() - 1
        mask = torch.randint(0, 2, (L, L))
        mask_type = 0  # LxL => src_mask
        input = input.mlu()
        mask = mask.mlu()
        mask = mask.bool()
        native_res = torch._masked_softmax(input, mask, dim, mask_type)
        mask = mask.expand(B, num_heads, L, L)
        mask = ~mask

        pt_res = self._slow_masked_softmax(input, mask, dim=dim)
        self.assertEqual(pt_res, native_res, exact_dtype=True)

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_masked_softmax_bfloat16(self):
        sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
        for B, num_heads, L in sizes:
            src_mask = torch.randint(0, 2, (L, L)).bool()

            src_key_padding_mask = torch.randint(0, 2, (B, L)).bool()

            generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()

            masks = [(src_mask, 0), (src_key_padding_mask, 1), (generic_mask, 2)]

            dims = [-1, 3]
            dtypes = [torch.bfloat16]
            for dim, dtype in product(dims, dtypes):
                input = torch.randn((B, num_heads, L, L), dtype=dtype)

                for mask, mask_type in masks:

                    def softmax_on_device(mask, input, device):
                        input_device = input.to(device)
                        mask_device = mask.to(device)
                        softmax_res = torch._masked_softmax(
                            input_device, mask_device, dim, mask_type
                        )
                        if mask_type == 0:
                            mask_expanded = (
                                mask_device.reshape(1, 1, L, L)
                                .expand(B, num_heads, L, L)
                                .bool()
                            )
                        elif mask_type == 1:
                            mask_expanded = (
                                mask_device.reshape(B, 1, 1, L)
                                .expand(B, num_heads, L, L)
                                .bool()
                            )
                        else:
                            mask_expanded = mask_device
                        mask_out = mask_expanded.all(dim, keepdim=True).expand(
                            mask_expanded.shape
                        )
                        softmax_res = softmax_res.masked_fill(mask_out, 0)
                        return softmax_res

                    cpu_res = softmax_on_device(mask, input.float(), "cpu")
                    mlu_res = softmax_on_device(mask, input, "mlu")
                    self.assertTensorsEqual(
                        cpu_res, mlu_res.cpu().float(), 0.003, use_MSE=True
                    )


if __name__ == "__main__":
    run_tests()
