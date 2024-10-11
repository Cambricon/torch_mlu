from __future__ import print_function

import sys
import os
import unittest
import logging
import math
import torch
import torch_mlu
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
from functools import partial
from torch.backends.mlu import sdp_kernel, SDPBackend
from itertools import product
from einops import rearrange, repeat

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    read_card_info,
)  # pylint: disable=C0413,C0411

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)
device = "mlu"
backend_map = {
    SDPBackend.MATH: {
        "enable_math": True,
        "enable_flash": False,
        "enable_mem_efficient": False,
    },
    SDPBackend.FLASH_ATTENTION: {
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": False,
    },
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False,
        "enable_flash": False,
        "enable_mem_efficient": True,
    },
}


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    scale=None,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Args:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.

    Returns:
        output: (batch_size, nheads, seqlen_q, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    scale = math.sqrt(d) if scale is None else (1 / scale)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / scale)
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og).transpose(1, 2), attention.to(dtype=dtype_og)


default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    if deviation.dtype == torch.bfloat16:
        torch.nan_to_num_(deviation.cpu(), nan=default_rtol[computed_value.dtype])
        deviation = deviation.mlu()
    else:
        torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()


def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = 2,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    atol = get_atol(true_value, computed_value)
    rtol = get_rtol(true_value, computed_value)

    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol


def rand_sdpa_tensor(
    shape: Tuple[Union[int, List[int]]],
    device: str,
    dtype: torch.dtype,
    requires_grad: bool = False,
    packed: bool = False,
) -> torch.Tensor:
    """Creates rand dense tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    batch, seq_len, num_heads, head_dim = shape
    size = (
        (batch, seq_len, num_heads, head_dim)
        if not packed
        else (batch, seq_len, 3 * num_heads * head_dim)
    )
    return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)


class TestSDPAThrowException(TestCase):
    # 1.test check_runtime_disabled_backend exception in sdp_utils
    @testinfo()
    def test_dispatch_no_backend_exception(self):
        dtype = torch.float16
        with sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=False
        ):
            size = (2, 3, 4)
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(
                RuntimeError,
                "No viable backend for scaled_dot_product_attention was found.",
                lambda: torch._fused_sdp_choice(q, k, v),
            )
            self.assertRaisesRegex(
                RuntimeError,
                "No viable backend for scaled_dot_product_attention was found.",
                lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            )

    # 2.test check_tensor_shapes exception in sdp_utils
    @testinfo()
    def test_invalid_inputs_dim_3_exception(self):
        for kernel in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]:
            with sdp_kernel(**backend_map[kernel]):
                # Dim is not 4
                size = (2, 3, 8)
                dtype = torch.float16
                q = torch.randn(size, device=device, dtype=dtype)
                k = torch.randn(size, device=device, dtype=dtype)
                v = torch.randn(size, device=device, dtype=dtype)
                with self.assertWarnsRegex(
                    UserWarning,
                    "Both fused kernels requires query, key and value to be 4 dimensional",
                ):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, None, 0.0, False
                        ),
                    )

    # 3.test check_batch_size_and_num_heads exception in sdp_utils
    @testinfo()
    def test_invalid_inputs_broadcast_exception(self):
        for kernel in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]:
            with sdp_kernel(**backend_map[kernel]):
                dtype = torch.float16
                size = (2, 16, 512, 256)
                size_broadcast = (1, 16, 512, 256)
                q = torch.randn(size_broadcast, device=device, dtype=dtype)
                k = torch.randn(size, device=device, dtype=dtype)
                v = torch.randn(size, device=device, dtype=dtype)
                with self.assertWarnsRegex(
                    UserWarning,
                    "For dense inputs, both fused kernels require query, key and value to have the same batch_size and num_heads.",
                ):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, None, 0.0, False
                        ),
                    )

    # 4.test check_for_attn_mask exception in sdp_utils
    @testinfo()
    def test_invalid_fused_inputs_attn_mask_present(self):
        for kernel in [SDPBackend.FLASH_ATTENTION]:
            with sdp_kernel(**backend_map[kernel]):
                # Failures for unsupported SDP args
                size = (2, 2, 3, 16)
                make_tensor = partial(
                    rand_sdpa_tensor, device=device, dtype=torch.float16
                )
                q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
                # Non-None attention mask
                mask = torch.ones((2, 2, 3, 3), device=device, dtype=q.dtype)
                with self.assertWarnsRegex(
                    UserWarning, "Flash Attention does not support non-null attn_mask."
                ):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, mask, 0.0, False
                        ),
                    )

    # 5.test check_head_dim_size（include check_head_dim_size_mem_efficient）exception in sdp_utils
    @testinfo()
    def test_invalid_fused_inputs_head_dim_exception(self):
        for kernel in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]:
            with sdp_kernel(**backend_map[kernel]):
                dtype = torch.float16
                make_tensor = partial(rand_sdpa_tensor, device=device, dtype=dtype)
                size_q = (2, 2, 3, 9)
                size_k = (2, 2, 3, 10)
                q, k, v = make_tensor(size_q), make_tensor(size_k), make_tensor(size_k)
                with self.assertWarns(UserWarning):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, None, 0.0, False
                        ),
                    )

    # 6.test check_nonzero_sequence_lengths exception in sdp_utils
    @testinfo()
    def test_invalid_sequence_lengths_exception(self):
        for kernel in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]:
            with sdp_kernel(**backend_map[kernel]):
                # Passing in a q,k,v with 0 length sequences will error
                dtype = torch.float16
                make_tensor = partial(rand_sdpa_tensor, device=device, dtype=dtype)
                size = (2, 2, 0, 128)
                q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
                with self.assertWarnsRegex(
                    UserWarning,
                    "Both fused kernels do not support zero seq_len_q or seq_len_kv.",
                ):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, None, 0.0, False
                        ),
                    )

    # 7.test_last_dim_stride_equals_1 exception in sdp_utils
    @testinfo()
    def test_invalid_last_dim_stride_exception(self):
        for kernel in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]:
            with sdp_kernel(**backend_map[kernel]):
                dtype = torch.float16
                make_tensor = partial(rand_sdpa_tensor, device=device, dtype=dtype)
                size = (2, 2, 8, 128)
                q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
                q.as_strided_(size, [2, 2, 2, 2])
                with self.assertWarnsRegex(
                    UserWarning,
                    "Both fused kernels require the last dimension of the input to have stride 1.",
                ):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, None, 0.0, False
                        ),
                    )

    # 8.test check_tesor_dtype exception in sdp_utils
    @testinfo()
    def test_invalid_fused_inputs_dtype_exception(self):
        backend_list = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        type_list = [torch.float, torch.float64]
        for kernel, dtype in product(backend_list, type_list):
            with sdp_kernel(**backend_map[kernel]):
                # Invalid dtype for both Flash Attention and Mem Efficient Attention
                size = (2, 16, 512, 256)
                make_tensor = partial(rand_sdpa_tensor, device=device, dtype=dtype)
                q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
                with self.assertWarns(UserWarning):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, None, 0.0, False
                        ),
                    )

    # 9.test check_fused_kernel_mlu_support exception in sdp_utils
    @testinfo()
    @unittest.skipUnless((not read_card_info()), "Dont test on selected MLU series")
    def test_mlu_arch_exception(self):
        dtype = torch.float16
        size = (2, 2, 3, 256)
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=dtype)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        backend_list = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        for kernel in backend_list:
            with sdp_kernel(**backend_map[kernel]):
                with self.assertWarnsRegex(
                    UserWarning,
                    "Both fused kernels only supports specified MLU series.",
                ):
                    self.assertRaises(
                        RuntimeError,
                        lambda: torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, None, 0.0, False
                        ),
                    )


class TestSDPA(TestCase):
    def query_key_value_clones(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype,
    ):
        """Clones the query, key, and value tensors and moves them to the specified dtype."""
        query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
        key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
        value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
        return query_ref, key_ref, value_ref

    @testinfo()
    def test_scaled_dot_product_attention_math_with_negative_scale(self):
        def ref(x):
            query = x * (-100)
            v1 = torch.matmul(query, x.transpose(-1, -2) * 100)
            v2 = v1.softmax(dim=-1)
            v3 = torch.matmul(v2, x)
            return v3

        x = torch.randn(1, 3, 64, 64, device=device)
        ref_result = ref(x)
        with sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            sdp_math = torch.nn.functional.scaled_dot_product_attention(
                x, x, x, scale=-1.0 / 0.0001
            )
        self.assertEqual(ref_result, sdp_math)

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_fused_sdp_choice(self):
        batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 128
        shape = (batch_size, seq_len, num_heads, head_dim)
        make_tensor = partial(
            rand_sdpa_tensor,
            device=device,
            dtype=torch.float16,
            packed=True,
            requires_grad=True,
        )
        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        for has_attn_mask in [True, False]:
            if not has_attn_mask:
                assert (
                    torch._fused_sdp_choice(query, key, value)
                    == SDPBackend.FLASH_ATTENTION.value
                )
            else:
                attn_mask = torch.rand(
                    seq_len, seq_len, device=device, dtype=torch.float16
                )
                assert (
                    torch._fused_sdp_choice(query, key, value, attn_mask=attn_mask)
                    == SDPBackend.EFFICIENT_ATTENTION.value
                )

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_flash_autocast_fp32(self):
        dtype = torch.float
        shape = (16, 16, 512, 128)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        with torch.autocast(device_type=device, dtype=torch.float16):
            with sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            ):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False
                )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_flash_autocast_fp32_bfloat16(self):
        dtype = torch.float
        shape = (16, 16, 512, 128)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            with sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            ):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False
                )

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_scaled_dot_product_attention_fused_kernels(self):
        is_contiguous_list = [True, False]
        is_packed_list = [True, False]

        for is_contiguous, is_packed in product(is_contiguous_list, is_packed_list):
            make_tensor = partial(
                rand_sdpa_tensor, device=device, dtype=torch.float16, packed=is_packed
            )

            batch_size, seq_len, num_heads, head_dim = 32, 512, 4, 128

            shape = (batch_size, seq_len, num_heads, head_dim)

            if is_packed:
                qkv = make_tensor(shape)
                query, key, value = qkv.chunk(3, dim=-1)
            else:
                query = make_tensor(shape)
                key = make_tensor(shape)
                value = make_tensor(shape)

            # Lets switch seq_len and num_heads
            # B x S X H X D -> B x H x S x D
            query = query.view(batch_size, -1, num_heads, head_dim)
            key = key.view(batch_size, -1, num_heads, head_dim)
            value = value.view(batch_size, -1, num_heads, head_dim)
            query_t = query.transpose(1, 2)
            key_t = key.transpose(1, 2)
            value_t = value.transpose(1, 2)

            if is_contiguous:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

            with sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                actual_memory = torch.nn.functional.scaled_dot_product_attention(
                    query_t,
                    key_t,
                    value_t,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

            with sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                actual_flash = torch.nn.functional.scaled_dot_product_attention(
                    query_t,
                    key_t,
                    value_t,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

            out_lp_ref, _ = attention_ref(query, key, value, upcast=False)
            out_ref, _ = attention_ref(query, key, value, upcast=True)
            output_atol, output_rtol = get_tolerances(out_ref, out_lp_ref)
            self.assertEqual(
                actual_flash,
                out_ref.to(actual_flash.dtype),
                atol=output_atol,
                rtol=output_rtol,
            )
            self.assertEqual(
                actual_memory,
                out_ref.to(actual_memory.dtype),
                atol=output_atol,
                rtol=output_rtol,
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_scaled_dot_product_attention_fused_kernels_bfloat16(self):
        is_contiguous_list = [True, False]
        is_packed_list = [True, False]

        for is_contiguous, is_packed in product(is_contiguous_list, is_packed_list):
            make_tensor = partial(
                rand_sdpa_tensor, device=device, dtype=torch.bfloat16, packed=is_packed
            )

            batch_size, seq_len, num_heads, head_dim = 1, 512, 16, 128

            shape = (batch_size, seq_len, num_heads, head_dim)

            if is_packed:
                qkv = make_tensor(shape)
                query, key, value = qkv.chunk(3, dim=-1)
            else:
                query = make_tensor(shape)
                key = make_tensor(shape)
                value = make_tensor(shape)

            # Lets switch seq_len and num_heads
            # B x S X H X D -> B x H x S x D
            query = query.view(batch_size, -1, num_heads, head_dim)
            key = key.view(batch_size, -1, num_heads, head_dim)
            value = value.view(batch_size, -1, num_heads, head_dim)
            query_t = query.transpose(1, 2)
            key_t = key.transpose(1, 2)
            value_t = value.transpose(1, 2)

            if is_contiguous:
                query = query.contiguous()
                key = key.contiguous()
                value = value.contiguous()

            with sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                actual_memory = torch.nn.functional.scaled_dot_product_attention(
                    query_t,
                    key_t,
                    value_t,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

            with sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                actual_flash = torch.nn.functional.scaled_dot_product_attention(
                    query_t,
                    key_t,
                    value_t,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )

            out_lp_ref, _ = attention_ref(query, key, value, upcast=False)
            out_ref, _ = attention_ref(query, key, value, upcast=True)

            output_atol, output_rtol = get_tolerances(out_ref, out_lp_ref)
            self.assertEqual(actual_flash, out_ref, atol=output_atol, rtol=output_rtol)
            self.assertEqual(actual_memory, out_ref, atol=output_atol, rtol=output_rtol)

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_scaled_dot_product_attention_fused_kernels_backward(self):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(
            rand_sdpa_tensor,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
            packed=True,
        )
        qkv_lp_fa = make_tensor((batch_size, seq_len, num_heads, head_dim))
        qkv_lp_me = qkv_lp_fa.detach().clone().requires_grad_()
        qkv_ref = qkv_lp_fa.detach().clone().to(torch.float64).requires_grad_()
        qkv_lp_ref = qkv_lp_fa.detach().clone().requires_grad_()

        query_fa, key_fa, value_fa = qkv_lp_fa.chunk(3, dim=-1)
        query_me, key_me, value_me = qkv_lp_me.chunk(3, dim=-1)
        query_ref, key_ref, value_ref = qkv_ref.chunk(3, dim=-1)
        query_lp_ref, key_lp_ref, value_lp_ref = qkv_lp_ref.chunk(3, dim=-1)

        query_fa = query_fa.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_fa = key_fa.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_fa = value_fa.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_me = query_me.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_me = key_me.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_me = value_me.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_ref = query_ref.view(batch_size, -1, num_heads, head_dim)
        key_ref = key_ref.view(batch_size, -1, num_heads, head_dim)
        value_ref = value_ref.view(batch_size, -1, num_heads, head_dim)

        query_lp_ref = query_lp_ref.view(batch_size, -1, num_heads, head_dim)
        key_lp_ref = key_lp_ref.view(batch_size, -1, num_heads, head_dim)
        value_lp_ref = value_lp_ref.view(batch_size, -1, num_heads, head_dim)

        contiguous_list = [True, False]
        causal_list = [True, False]
        for is_contiguous, is_causal in product(contiguous_list, causal_list):
            if is_contiguous:
                query_fa = query_fa.contiguous()
                key_fa = key_fa.contiguous()
                value_fa = value_fa.contiguous()

                query_me = query_me.contiguous()
                key_me = key_me.contiguous()
                value_me = value_me.contiguous()

                query_lp_ref = query_lp_ref.contiguous()
                key_lp_ref = key_lp_ref.contiguous()
                value_lp_ref = value_lp_ref.contiguous()

                query_ref = query_ref.contiguous()
                key_ref = key_ref.contiguous()
                value_ref = value_ref.contiguous()

            with sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                actual_memory = torch.nn.functional.scaled_dot_product_attention(
                    query_me,
                    key_me,
                    value_me,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )

            with sdp_kernel(
                enable_math=False, enable_mem_efficient=False, enable_flash=True
            ):
                actual_flash = torch.nn.functional.scaled_dot_product_attention(
                    query_fa, key_fa, value_fa, None, 0.0, is_causal
                )

            out_lp_ref, _ = attention_ref(
                query_lp_ref, key_lp_ref, value_lp_ref, causal=is_causal, upcast=False
            )
            out_ref, _ = attention_ref(
                query_ref, key_ref, value_ref, causal=is_causal, upcast=False
            )
            rand_upward = torch.rand_like(out_ref)
            rand_upward_lp = rand_upward.to(torch.float16)
            actual_flash.backward(rand_upward_lp)
            actual_memory.backward(rand_upward_lp)
            out_ref.backward(rand_upward)
            out_lp_ref.backward(rand_upward_lp)

            grad_qkv_atol, grad_kev_rtol = get_tolerances(qkv_ref.grad, qkv_lp_ref.grad)
            self.assertEqual(
                qkv_ref.grad,
                qkv_lp_fa.grad.to(torch.float64),
                atol=grad_qkv_atol,
                rtol=grad_kev_rtol,
            )
            self.assertEqual(
                qkv_ref.grad,
                qkv_lp_me.grad.to(torch.float64),
                atol=grad_qkv_atol,
                rtol=grad_kev_rtol,
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_scaled_dot_product_attention_fused_kernels_backward_bfloat16(self):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(
            rand_sdpa_tensor,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
            packed=True,
        )
        qkv_lp_fa = make_tensor((batch_size, seq_len, num_heads, head_dim))
        qkv_lp_me = qkv_lp_fa.detach().clone().requires_grad_()
        qkv_ref = qkv_lp_fa.detach().clone().to(torch.float64).requires_grad_()
        qkv_lp_ref = qkv_lp_fa.detach().clone().requires_grad_()

        query_fa, key_fa, value_fa = qkv_lp_fa.chunk(3, dim=-1)
        query_me, key_me, value_me = qkv_lp_me.chunk(3, dim=-1)
        query_ref, key_ref, value_ref = qkv_ref.chunk(3, dim=-1)
        query_lp_ref, key_lp_ref, value_lp_ref = qkv_lp_ref.chunk(3, dim=-1)

        query_fa = query_fa.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_fa = key_fa.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_fa = value_fa.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_me = query_me.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_me = key_me.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_me = value_me.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_ref = query_ref.view(batch_size, -1, num_heads, head_dim)
        key_ref = key_ref.view(batch_size, -1, num_heads, head_dim)
        value_ref = value_ref.view(batch_size, -1, num_heads, head_dim)

        query_lp_ref = query_lp_ref.view(batch_size, -1, num_heads, head_dim)
        key_lp_ref = key_lp_ref.view(batch_size, -1, num_heads, head_dim)
        value_lp_ref = value_lp_ref.view(batch_size, -1, num_heads, head_dim)

        contiguous_list = [True, False]
        causal_list = [True, False]
        for is_contiguous, is_causal in product(contiguous_list, causal_list):
            if is_contiguous:
                query_fa = query_fa.contiguous()
                key_fa = key_fa.contiguous()
                value_fa = value_fa.contiguous()

                query_me = query_me.contiguous()
                key_me = key_me.contiguous()
                value_me = value_me.contiguous()

                query_lp_ref = query_lp_ref.contiguous()
                key_lp_ref = key_lp_ref.contiguous()
                value_lp_ref = value_lp_ref.contiguous()

                query_ref = query_ref.contiguous()
                key_ref = key_ref.contiguous()
                value_ref = value_ref.contiguous()

            with sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ):
                actual_memory = torch.nn.functional.scaled_dot_product_attention(
                    query_me,
                    key_me,
                    value_me,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )

            with sdp_kernel(
                enable_math=False, enable_mem_efficient=False, enable_flash=True
            ):
                actual_flash = torch.nn.functional.scaled_dot_product_attention(
                    query_fa, key_fa, value_fa, None, 0.0, is_causal
                )

            out_lp_ref, _ = attention_ref(
                query_lp_ref, key_lp_ref, value_lp_ref, causal=is_causal, upcast=False
            )
            out_ref, _ = attention_ref(
                query_ref, key_ref, value_ref, causal=is_causal, upcast=False
            )

            rand_upward = torch.rand_like(out_ref)
            rand_upward_lp = rand_upward.mlu().to(torch.float16)
            actual_flash.backward(rand_upward_lp)
            actual_memory.backward(rand_upward_lp)
            out_ref.backward(rand_upward)
            out_lp_ref.backward(rand_upward_lp)

            grad_qkv_atol, grad_kev_rtol = get_tolerances(qkv_ref.grad, qkv_lp_ref.grad)
            self.assertEqual(
                qkv_ref.grad,
                qkv_lp_fa.grad.to(torch.float64),
                atol=grad_qkv_atol,
                rtol=grad_kev_rtol,
            )
            self.assertEqual(
                qkv_ref.grad,
                qkv_lp_me.grad.to(torch.float64),
                atol=grad_qkv_atol,
                rtol=grad_kev_rtol,
            )

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_scaled_dot_product_flash_attention_backward(self):
        batch_size_list = [1, 8]
        seq_len_q_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        seq_len_k_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        head_dim_list = [8, 16, 32, 64, 72, 96, 128]
        scale_list = [None, "l1"]
        is_causal_list = [True, False]
        dropout_p_list = [0.0, 0.22]
        dtype_list = [torch.float16]
        seed = 42

        for (
            batch_size,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            dropout_p,
            dtype,
        ) in product(
            batch_size_list,
            seq_len_q_list,
            seq_len_k_list,
            head_dim_list,
            scale_list,
            is_causal_list,
            dropout_p_list,
            dtype_list,
        ):
            n_heads = 4
            scale = scale if scale is None else (1 / head_dim)

            query = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )

            query_lp_ref, key_lp_ref, value_lp_ref = self.query_key_value_clones(
                query, key, value, dtype=dtype
            )

            higher_precision_dtype = torch.float32
            query_ref, key_ref, value_ref = self.query_key_value_clones(
                query, key, value, dtype=higher_precision_dtype
            )

            # Create real output
            torch.manual_seed(seed)
            output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                return_debug_mask=True,
            )
            out = output_tuple[0]

            with sdp_kernel(
                enable_math=True, enable_flash=False, enable_mem_efficient=False
            ):
                torch.manual_seed(seed)
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(
                    query_ref,
                    key_ref,
                    value_ref,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query_lp_ref,
                    key_lp_ref,
                    value_lp_ref,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )

            upstream_grad = torch.rand_like(out, requires_grad=False)
            out.backward(upstream_grad)
            out_ref.backward(upstream_grad.to(out_ref.dtype))
            out_lp_ref.backward(upstream_grad)

            output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
            # TODO: Investigate why grad_q needs larger tolerances
            query_fudge_factor = 4
            grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
                query_ref.grad, query_lp_ref.grad, query_fudge_factor
            )
            grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(
                key_ref.grad, key_lp_ref.grad
            )
            grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(
                value_ref.grad, value_lp_ref.grad
            )

            self.assertEqual(
                out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol
            )
            self.assertEqual(
                query.grad,
                query_ref.grad.to(query.grad.dtype),
                atol=grad_q_ref_atol,
                rtol=grad_q_ref_rtol,
            )
            self.assertEqual(
                key.grad,
                key_ref.grad.to(key.grad.dtype),
                atol=grad_k_ref_atol,
                rtol=grad_k_ref_rtol,
            )
            self.assertEqual(
                value.grad,
                value_ref.grad.to(value.grad.dtype),
                atol=grad_v_ref_atol,
                rtol=grad_v_ref_rtol,
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_scaled_dot_product_flash_attention_backward_bfloat16(self):
        batch_size_list = [1, 8]
        seq_len_q_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        seq_len_k_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        head_dim_list = [8, 16, 32, 64, 72, 96, 128]
        scale_list = [None, "l1"]
        is_causal_list = [True, False]
        dropout_p_list = [0.0, 0.22]
        dtype_list = [torch.bfloat16]
        seed = 42
        for (
            batch_size,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            dropout_p,
            dtype,
        ) in product(
            batch_size_list,
            seq_len_q_list,
            seq_len_k_list,
            head_dim_list,
            scale_list,
            is_causal_list,
            dropout_p_list,
            dtype_list,
        ):
            n_heads = 4
            scale = scale if scale is None else (1 / head_dim)

            query = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )

            query_lp_ref, key_lp_ref, value_lp_ref = self.query_key_value_clones(
                query, key, value, dtype=dtype
            )

            higher_precision_dtype = torch.float32
            query_ref, key_ref, value_ref = self.query_key_value_clones(
                query, key, value, dtype=higher_precision_dtype
            )

            # Create real output
            torch.manual_seed(seed)
            output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                return_debug_mask=True,
            )
            out = output_tuple[0]

            with sdp_kernel(
                enable_math=True, enable_flash=False, enable_mem_efficient=False
            ):
                torch.manual_seed(seed)
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(
                    query_ref,
                    key_ref,
                    value_ref,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query_lp_ref,
                    key_lp_ref,
                    value_lp_ref,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )

            upstream_grad = torch.rand_like(out, requires_grad=False)
            out.backward(upstream_grad)
            out_ref.backward(upstream_grad.to(out_ref.dtype))
            out_lp_ref.backward(upstream_grad)

            output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
            query_fudge_factor = 4
            grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
                query_ref.grad, query_lp_ref.grad, query_fudge_factor
            )
            grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(
                key_ref.grad, key_lp_ref.grad
            )
            grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(
                value_ref.grad, value_lp_ref.grad
            )

            self.assertEqual(
                out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol
            )
            self.assertEqual(
                query.grad,
                query_ref.grad.to(query.grad.dtype),
                atol=grad_q_ref_atol,
                rtol=grad_q_ref_rtol,
            )
            self.assertEqual(
                key.grad,
                key_ref.grad.to(key.grad.dtype),
                atol=grad_k_ref_atol,
                rtol=grad_k_ref_rtol,
            )
            self.assertEqual(
                value.grad,
                value_ref.grad.to(value.grad.dtype),
                atol=grad_v_ref_atol,
                rtol=grad_v_ref_rtol,
            )

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_mem_efficient_attetntion_mask_variants(self):
        dtype = torch.float16
        make_tensor = partial(
            rand_sdpa_tensor, device=device, dtype=dtype, requires_grad=True
        )
        batch, num_heads, head_dim = 6, 6, 128
        seq_len_q, seq_len_kv = 512, 512

        query = make_tensor((batch, num_heads, seq_len_q, head_dim))
        kv_shape = (batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)

        for mask_dim in [1, 2, 3, 4]:
            if mask_dim == 1:
                mask = torch.randn((seq_len_kv,), device=device, dtype=dtype)
            elif mask_dim == 2:
                mask = torch.randn((seq_len_q, seq_len_kv), device=device, dtype=dtype)
            elif mask_dim == 3:
                mask = torch.randn(
                    (num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype
                )
            elif mask_dim == 4:
                mask = torch.randn(
                    (batch, num_heads, seq_len_q, seq_len_kv),
                    device=device,
                    dtype=dtype,
                )
            with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(query, key, value, mask)
            out.sum().backward()

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_mem_eff_attention_non_contiguous_mask(self):
        dtype = torch.float16
        make_tensor = partial(
            rand_sdpa_tensor, device=device, dtype=dtype, requires_grad=True
        )
        batch, num_heads, head_dim = 8, 8, 128
        seq_len_q, seq_len_kv = 512, 1024

        query = make_tensor((batch, num_heads, seq_len_q, head_dim))
        kv_shape = (batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)

        mask = torch.randn(
            (batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype
        )
        mask = torch.as_strided(
            mask, (batch, num_heads, seq_len_q, seq_len_kv), (0, 0, 0, 1)
        )
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_mem_eff_attention_long_sequence_mask(self):
        dtype = torch.float16
        make_tensor = partial(
            rand_sdpa_tensor, device=device, dtype=dtype, requires_grad=True
        )
        batch, num_heads, head_dim = 1, 32, 128
        seq_len_q, seq_len_kv = 8192, 8192

        query = make_tensor((batch, num_heads, seq_len_q, head_dim))
        kv_shape = (batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)

        mask = torch.randn(
            (batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype
        )
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_mem_efficient_attention_vs_math_ref_grads(self):
        batch_size_list = [1, 8]
        seq_len_q_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        seq_len_k_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        head_dim_list = [8, 16, 32, 64, 72, 96, 128]
        scale_list = [None, "l1"]
        is_causal_list = [False, True]
        dropout_p_list = [0.0, 0.22]
        dtype_list = [torch.float16]

        seed = 42
        for (
            batch_size,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            dropout_p,
            dtype,
        ) in product(
            batch_size_list,
            seq_len_q_list,
            seq_len_k_list,
            head_dim_list,
            scale_list,
            is_causal_list,
            dropout_p_list,
            dtype_list,
        ):
            n_heads = 4
            scale = scale if scale is None else (1 / head_dim)

            query = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )

            query_lp_ref, key_lp_ref, value_lp_ref = self.query_key_value_clones(
                query, key, value, dtype=dtype
            )

            higher_precision_dtype = torch.float32
            query_ref, key_ref, value_ref = self.query_key_value_clones(
                query, key, value, dtype=higher_precision_dtype
            )

            # Create real output
            with sdp_kernel(
                enable_mem_efficient=True, enable_flash=False, enable_math=False
            ):
                torch.manual_seed(seed)
                out = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )

            if dropout_p == 0.0:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                    )
            else:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    torch.manual_seed(seed)
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        dropout_p=dropout_p,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        dropout_p=dropout_p,
                    )

            upstream_grad = torch.rand_like(out, requires_grad=False)
            out.backward(upstream_grad)
            out_ref.backward(upstream_grad.to(out_ref.dtype))
            out_lp_ref.backward(upstream_grad)
            output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
            query_fudge_factor = 4
            grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
                query_ref.grad, query_lp_ref.grad, query_fudge_factor
            )
            grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(
                key_ref.grad, key_lp_ref.grad
            )
            grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(
                value_ref.grad, value_lp_ref.grad
            )

            self.assertEqual(
                out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol
            )
            self.assertEqual(
                query.grad,
                query_ref.grad.to(query.grad.dtype),
                atol=grad_q_ref_atol,
                rtol=grad_q_ref_rtol,
            )
            self.assertEqual(
                key.grad,
                key_ref.grad.to(key.grad.dtype),
                atol=grad_k_ref_atol,
                rtol=grad_k_ref_rtol,
            )
            self.assertEqual(
                value.grad,
                value_ref.grad.to(value.grad.dtype),
                atol=grad_v_ref_atol,
                rtol=grad_v_ref_rtol,
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_mem_efficient_attention_vs_math_ref_grads_bfloat16(self):
        batch_size_list = [1, 8]
        seq_len_q_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        seq_len_k_list = [4, 8, 64, 128, 256, 512, 1024, 2048]
        head_dim_list = [8, 16, 32, 64, 72, 96, 128]
        scale_list = [None, "l1"]
        is_causal_list = [False, True]
        dropout_p_list = [0.0, 0.22]
        dtype_list = [torch.bfloat16]

        seed = 42
        for (
            batch_size,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            dropout_p,
            dtype,
        ) in product(
            batch_size_list,
            seq_len_q_list,
            seq_len_k_list,
            head_dim_list,
            scale_list,
            is_causal_list,
            dropout_p_list,
            dtype_list,
        ):
            n_heads = 4
            scale = scale if scale is None else (1 / head_dim)

            query = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )

            query_lp_ref, key_lp_ref, value_lp_ref = self.query_key_value_clones(
                query, key, value, dtype=dtype
            )

            higher_precision_dtype = torch.float32
            query_ref, key_ref, value_ref = self.query_key_value_clones(
                query, key, value, dtype=higher_precision_dtype
            )

            # Create real output
            with sdp_kernel(
                enable_mem_efficient=True, enable_flash=False, enable_math=False
            ):
                torch.manual_seed(seed)
                out = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )

            if dropout_p == 0.0:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                    )
            else:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    torch.manual_seed(seed)
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        dropout_p=dropout_p,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        dropout_p=dropout_p,
                    )

            upstream_grad = torch.rand_like(out, requires_grad=False)
            out.backward(upstream_grad)
            out_ref.backward(upstream_grad.to(out_ref.dtype))
            out_lp_ref.backward(upstream_grad)
            output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
            query_fudge_factor = 4
            grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
                query_ref.grad, query_lp_ref.grad, query_fudge_factor
            )
            grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(
                key_ref.grad, key_lp_ref.grad
            )
            grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(
                value_ref.grad, value_lp_ref.grad
            )

            self.assertEqual(
                out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol
            )
            self.assertEqual(
                query.grad,
                query_ref.grad.to(query.grad.dtype),
                atol=grad_q_ref_atol,
                rtol=grad_q_ref_rtol,
            )
            self.assertEqual(
                key.grad,
                key_ref.grad.to(key.grad.dtype),
                atol=grad_k_ref_atol,
                rtol=grad_k_ref_rtol,
            )
            self.assertEqual(
                value.grad,
                value_ref.grad.to(value.grad.dtype),
                atol=grad_v_ref_atol,
                rtol=grad_v_ref_rtol,
            )

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_mem_efficient_attention_attn_mask_vs_math_ref_grads(self):
        batch_size_list = [1, 8]
        seq_len_q_list = [4, 8, 64, 128, 256, 312, 512, 1024, 2048]
        seq_len_k_list = [4, 8, 64, 65, 128, 256, 408, 512, 1024, 2048]
        head_dim_list = [8, 16, 32, 64, 72, 96, 128]
        scale_list = [None, "l1"]
        is_causal_list = [False]
        dropout_p_list = [0.0, 0.22]
        dtype_list = [torch.float16]

        seed = 42
        for (
            batch_size,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            dropout_p,
            dtype,
        ) in product(
            batch_size_list,
            seq_len_q_list,
            seq_len_k_list,
            head_dim_list,
            scale_list,
            is_causal_list,
            dropout_p_list,
            dtype_list,
        ):
            n_heads = 4
            scale = scale if scale is None else (1 / head_dim)

            query = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )

            query_lp_ref, key_lp_ref, value_lp_ref = self.query_key_value_clones(
                query, key, value, dtype=dtype
            )

            higher_precision_dtype = torch.float32
            query_ref, key_ref, value_ref = self.query_key_value_clones(
                query, key, value, dtype=higher_precision_dtype
            )

            attn_mask_lp = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                seq_len_k,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            att_mask = attn_mask_lp.detach().clone().to(torch.float32)
            # Create real output
            with sdp_kernel(
                enable_mem_efficient=True, enable_flash=False, enable_math=False
            ):
                torch.manual_seed(seed)
                out = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask_lp,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )

            if dropout_p == 0.0:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=attn_mask_lp,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=att_mask,
                    )
            else:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    torch.manual_seed(seed)
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=attn_mask_lp,
                        dropout_p=dropout_p,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=att_mask,
                        dropout_p=dropout_p,
                    )

            upstream_grad = torch.rand_like(out, requires_grad=False)
            out.backward(upstream_grad)
            out_ref.backward(upstream_grad.to(out_ref.dtype))
            out_lp_ref.backward(upstream_grad)
            output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
            query_fudge_factor = 4
            grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
                query_ref.grad, query_lp_ref.grad, query_fudge_factor
            )
            grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(
                key_ref.grad, key_lp_ref.grad
            )
            grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(
                value_ref.grad, value_lp_ref.grad
            )

            self.assertEqual(
                out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol
            )
            self.assertEqual(
                query.grad,
                query_ref.grad.to(query.grad.dtype),
                atol=grad_q_ref_atol,
                rtol=grad_q_ref_rtol,
            )
            self.assertEqual(
                key.grad,
                key_ref.grad.to(key.grad.dtype),
                atol=grad_k_ref_atol,
                rtol=grad_k_ref_rtol,
            )
            self.assertEqual(
                value.grad,
                value_ref.grad.to(value.grad.dtype),
                atol=grad_v_ref_atol,
                rtol=grad_v_ref_rtol,
            )

    @testinfo()
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    def test_mem_efficient_attention_attn_mask_vs_math_ref_grads_bfloat16(self):
        batch_size_list = [1, 8]
        seq_len_q_list = [4, 8, 64, 128, 256, 312, 512, 1024, 2048]
        seq_len_k_list = [4, 8, 64, 65, 128, 256, 408, 512, 1024, 2048]
        head_dim_list = [8, 16, 32, 64, 72, 96, 128]
        scale_list = [None, "l1"]
        is_causal_list = [False]
        dropout_p_list = [0.0, 0.22]
        dtype_list = [torch.bfloat16]

        seed = 42
        for (
            batch_size,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            dropout_p,
            dtype,
        ) in product(
            batch_size_list,
            seq_len_q_list,
            seq_len_k_list,
            head_dim_list,
            scale_list,
            is_causal_list,
            dropout_p_list,
            dtype_list,
        ):
            n_heads = 4
            scale = scale if scale is None else (1 / head_dim)

            query = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )

            query_lp_ref, key_lp_ref, value_lp_ref = self.query_key_value_clones(
                query, key, value, dtype=dtype
            )

            higher_precision_dtype = torch.float32
            query_ref, key_ref, value_ref = self.query_key_value_clones(
                query, key, value, dtype=higher_precision_dtype
            )

            attn_mask_lp = torch.rand(
                seq_len_q, seq_len_k, device=device, dtype=dtype, requires_grad=True
            )
            att_mask = attn_mask_lp.detach().clone().to(torch.float32)
            # Create real output
            with sdp_kernel(
                enable_mem_efficient=True, enable_flash=False, enable_math=False
            ):
                torch.manual_seed(seed)
                out = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask_lp,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )

            if dropout_p == 0.0:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=attn_mask_lp,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=att_mask,
                    )
            else:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    torch.manual_seed(seed)
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=attn_mask_lp,
                        dropout_p=dropout_p,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=att_mask,
                        dropout_p=dropout_p,
                    )

            upstream_grad = torch.rand_like(out, requires_grad=False)
            out.backward(upstream_grad)
            out_ref.backward(upstream_grad.to(out_ref.dtype))
            out_lp_ref.backward(upstream_grad)
            output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
            query_fudge_factor = 4
            grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
                query_ref.grad, query_lp_ref.grad, query_fudge_factor
            )
            grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(
                key_ref.grad, key_lp_ref.grad
            )
            grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(
                value_ref.grad, value_lp_ref.grad
            )

            self.assertEqual(
                out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol
            )
            self.assertEqual(
                query.grad,
                query_ref.grad.to(query.grad.dtype),
                atol=grad_q_ref_atol,
                rtol=grad_q_ref_rtol,
            )
            self.assertEqual(
                key.grad,
                key_ref.grad.to(key.grad.dtype),
                atol=grad_k_ref_atol,
                rtol=grad_k_ref_rtol,
            )
            self.assertEqual(
                value.grad,
                value_ref.grad.to(value.grad.dtype),
                atol=grad_v_ref_atol,
                rtol=grad_v_ref_rtol,
            )

    @testinfo()
    @unittest.skipUnless(read_card_info(), "Only test on selected MLU series")
    def test_memory_efficient_attention(self):
        batch_size_list = [4, 8]
        seq_len_q_list = [512, 1024, 2048]
        seq_len_k_list = [512, 1024, 2048]
        head_dim_list = [128, 256]
        scale_list = [None]
        is_causal_list = [False, True]
        dropout_p_list = [0.0, 0.22, 0.48]
        dtype_list = [torch.float16]
        seed = 42
        for (
            batch_size,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            dropout_p,
            dtype,
        ) in product(
            batch_size_list,
            seq_len_q_list,
            seq_len_k_list,
            head_dim_list,
            scale_list,
            is_causal_list,
            dropout_p_list,
            dtype_list,
        ):
            n_heads = 4
            query = torch.rand(
                batch_size,
                n_heads,
                seq_len_q,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.rand(
                batch_size,
                n_heads,
                seq_len_k,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )

            query_lp_ref, key_lp_ref, value_lp_ref = self.query_key_value_clones(
                query, key, value, dtype=dtype
            )

            higher_precision_dtype = torch.float32
            query_ref, key_ref, value_ref = self.query_key_value_clones(
                query, key, value, dtype=higher_precision_dtype
            )
            attn_mask_lp = None
            att_mask = None
            attn_mask_expand = None
            if not is_causal:
                attn_mask_lp = torch.rand(
                    seq_len_q, seq_len_k, device=device, dtype=dtype, requires_grad=True
                )
                att_mask = attn_mask_lp.detach().clone().to(torch.float32)
                attn_mask_expand = attn_mask_lp.expand(
                    query.size(0), query.size(1), query.size(2), key.size(2)
                )

            cu_seqlens_q = torch.arange(
                0,
                (batch_size + 1) * seq_len_q,
                seq_len_q,
                device="mlu",
                dtype=torch.int,
            )
            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * seq_len_k,
                seq_len_k,
                device="mlu",
                dtype=torch.int,
            )
            query_t = query.transpose(1, 2)
            key_t = key.transpose(1, 2)
            value_t = value.transpose(1, 2)

            # Create real output
            torch.manual_seed(seed)
            (
                out,
                logsumexp,
                seed,
                offset,
            ) = torch.ops.aten._efficient_attention_forward(
                query_t,
                key_t,
                value_t,
                attn_mask_expand,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=None,
                dropout_p=dropout_p,
                custom_mask_type=1 if is_causal else 0,
                compute_log_sumexp=True,
            )

            if dropout_p == 0.0:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=attn_mask_lp,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=att_mask,
                    )
            else:
                with sdp_kernel(
                    enable_math=True, enable_flash=False, enable_mem_efficient=False
                ):
                    torch.manual_seed(seed)
                    out_lp_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_lp_ref,
                        key_lp_ref,
                        value_lp_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=attn_mask_lp,
                        dropout_p=dropout_p,
                    )
                    out_ref = torch.nn.functional.scaled_dot_product_attention(
                        query_ref,
                        key_ref,
                        value_ref,
                        is_causal=is_causal,
                        scale=scale,
                        attn_mask=att_mask,
                        dropout_p=dropout_p,
                    )

            upstream_grad = torch.rand_like(out, requires_grad=False)
            upstream_grad_t = upstream_grad.transpose(1, 2)

            out_t = out.transpose(1, 2)
            grad_q, grad_k, grad_v, _ = torch.ops.aten._efficient_attention_backward(
                upstream_grad_t,
                query_t,
                key_t,
                value_t,
                attn_mask_expand,
                out_t,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_k=seq_len_k,
                max_seqlen_q=seq_len_q,
                logsumexp=logsumexp,
                dropout_p=dropout_p,
                philox_seed=seed,
                philox_offset=offset,
                custom_mask_type=1 if is_causal else 0,
                bias_requires_grad=False,
            )
            grad_q_t = grad_q.transpose(1, 2)
            grad_k_t = grad_k.transpose(1, 2)
            grad_v_t = grad_v.transpose(1, 2)

            out_ref.backward(upstream_grad.to(out_ref.dtype))
            out_lp_ref.backward(upstream_grad)
            output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)

            query_fudge_factor = 4
            grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
                query_ref.grad, query_lp_ref.grad, query_fudge_factor
            )
            grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(
                key_ref.grad, key_lp_ref.grad
            )
            grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(
                value_ref.grad, value_lp_ref.grad
            )
            self.assertEqual(
                out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol
            )
            self.assertEqual(
                grad_q_t,
                query_ref.grad.to(grad_q_t.dtype),
                atol=grad_q_ref_atol,
                rtol=grad_q_ref_rtol,
            )
            self.assertEqual(
                grad_k_t,
                key_ref.grad.to(grad_k_t.dtype),
                atol=grad_k_ref_atol,
                rtol=grad_k_ref_rtol,
            )
            self.assertEqual(
                grad_v_t,
                value_ref.grad.to(grad_q_t.dtype),
                atol=grad_v_ref_atol,
                rtol=grad_v_ref_rtol,
            )


if __name__ == "__main__":
    unittest.main()
