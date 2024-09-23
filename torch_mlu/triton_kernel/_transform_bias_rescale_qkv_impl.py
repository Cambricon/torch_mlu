import math
from typing import Tuple

import torch
import torch_mlu

import triton
import triton.language as tl
from .libentry import libentry, get_total_core_num


# The split is done on the lowest dimension, so there is no limit on the qkv size.
def get_autotune_config():
    # YBLOCK, XBLOCK
    blocks_configs = [
        [1, 1],
        [1, 8],
        [1, 16],
        [1, 32],
    ]
    configs = [
        triton.Config(
            {
                "YBLOCK": y,
                "XBLOCK": x,
                "IS_DIVISIBLE": is_divisible,
            },
            num_stages=3,
            num_warps=1,
        )
        for y, x in blocks_configs
        for is_divisible in [True, False]
    ]
    return configs


# There must be at least two config to take effect
def config_prune(configs, named_args, **kwargs):
    T = named_args["T"]
    pruned_configs = []
    for config in configs:
        x = config.kwargs["XBLOCK"]
        is_divisible = config.kwargs["IS_DIVISIBLE"]
        if T % x == 0 and is_divisible:
            pruned_configs.append(config)
        elif T % x != 0 and not is_divisible:
            pruned_configs.append(config)
    return pruned_configs


@libentry()
@triton.autotune(
    configs=get_autotune_config(),
    key=["T", "D", "num_heads", "dim_per_head"],
    prune_configs_by={"early_config_prune": config_prune},
)
@triton.jit
def transform_bias_rescale_qkv_no_split_lowest_dim_impl(
    qkv_ptr,
    qkv_bias_ptr,
    out_q_ptr,
    out_k_ptr,
    out_v_ptr,
    inv_sqrt_dim_per_head,
    B,
    T: tl.constexpr,
    D: tl.constexpr,
    num_heads: tl.constexpr,
    dim_per_head: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
    IS_DIVISIBLE: tl.constexpr,
):
    qkv_bias_block_ptr = tl.make_block_ptr(
        qkv_bias_ptr,
        shape=[3 * D],
        strides=[1],
        block_shape=[3 * D],
        order=[0],
        offsets=[0],
    )
    qkv_bias = tl.load(
        qkv_bias_block_ptr,
    )[None, None, :]

    num_yblocks = tl.cdiv(B, YBLOCK)
    num_xblocks = tl.cdiv(T, XBLOCK)
    num_blocks = num_yblocks * num_xblocks

    pid = tl.program_id(0)
    block_start = pid
    step = tl.num_programs(0)

    for block_idx in range(block_start, num_blocks, step):
        yoffset = block_idx // num_xblocks * YBLOCK
        xoffset = block_idx % num_xblocks * XBLOCK
        qkv_block_ptr = tl.make_block_ptr(
            qkv_ptr,
            shape=[B, T, 3 * D],
            strides=[T * 3 * D, 3 * D, 1],
            block_shape=[YBLOCK, XBLOCK, 3 * D],
            order=[2, 1, 0],
            offsets=[yoffset, xoffset, 0],
        )
        qkv = tl.load(
            qkv_block_ptr,
            boundary_check=[0] if IS_DIVISIBLE else [0, 1],
            eviction_policy="evict_last",
        )
        tmp1 = qkv + qkv_bias
        tmp2 = tmp1[:, :, 0:D]
        tmp3 = tmp2 * inv_sqrt_dim_per_head.to(tmp2.dtype)
        tmp4 = tmp1[:, :, D : 2 * D]
        tmp5 = tmp1[:, :, 2 * D : 3 * D]

        tl.store(
            tl.make_block_ptr(
                out_q_ptr,
                shape=[B, num_heads, T, dim_per_head],
                strides=[T * D, T * dim_per_head, dim_per_head, 1],
                block_shape=[YBLOCK, num_heads, XBLOCK, dim_per_head],
                order=[3, 2, 1, 0],
                offsets=[yoffset, 0, xoffset, 0],
            ),
            tmp3.reshape([YBLOCK, XBLOCK, num_heads, dim_per_head]).trans(0, 2, 1, 3),
            boundary_check=[0] if IS_DIVISIBLE else [0, 2],
        )
        tl.store(
            tl.make_block_ptr(
                out_k_ptr,
                shape=[B, num_heads, T, dim_per_head],
                strides=[T * D, T * dim_per_head, dim_per_head, 1],
                block_shape=[YBLOCK, num_heads, XBLOCK, dim_per_head],
                order=[3, 2, 1, 0],
                offsets=[yoffset, 0, xoffset, 0],
            ),
            tmp4.reshape([YBLOCK, XBLOCK, num_heads, dim_per_head]).trans(0, 2, 1, 3),
            boundary_check=[0] if IS_DIVISIBLE else [0, 2],
        )
        tl.store(
            tl.make_block_ptr(
                out_v_ptr,
                shape=[B, num_heads, T, dim_per_head],
                strides=[T * D, T * dim_per_head, dim_per_head, 1],
                block_shape=[YBLOCK, num_heads, XBLOCK, dim_per_head],
                order=[3, 2, 1, 0],
                offsets=[yoffset, 0, xoffset, 0],
            ),
            tmp5.reshape([YBLOCK, XBLOCK, num_heads, dim_per_head]).trans(0, 2, 1, 3),
            boundary_check=[0] if IS_DIVISIBLE else [0, 2],
        )


def _transform_bias_rescale_qkv_mlu(
    qkv: torch.Tensor,
    qkv_bias: torch.Tensor,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert qkv.dim() == 3
    B = qkv.size(0)
    T = qkv.size(1)
    _3D = qkv.size(2)
    D = int(_3D / 3)
    dim_per_head = int(D / num_heads)
    assert D % num_heads == 0
    assert _3D % 3 == 0
    if not qkv.is_contiguous():
        qkv = qkv.clone()
    if not qkv_bias.is_contiguous():
        qkv_bias = qkv_bias.clone()
    dtype = qkv.dtype
    out_q = torch.empty((B, num_heads, T, dim_per_head), device=qkv.device, dtype=dtype)
    out_k = torch.empty((B, num_heads, T, dim_per_head), device=qkv.device, dtype=dtype)
    out_v = torch.empty((B, num_heads, T, dim_per_head), device=qkv.device, dtype=dtype)
    # 500 series:
    #     _3D must be less than or equal to 18432 when dtype is fp32
    #     _3D must be less than or equal to 32760 when dtype is fp16
    # 300 series:
    #     _3D must be less than or equal to 27960 when dtype is fp32
    #     _3D must be less than or equal to 32760 when dtype is fp16
    if (dtype == torch.float32 and _3D > 18432) or (
        dtype == torch.float16 and _3D > 32760
    ):
        raise RuntimeError(
            "_transform_bias_rescale_qkv: MLU does not support qkv.size(2) exceed {} for {}.".format(
                _3D, dtype
            )
        )
    if qkv.numel() > 0:
        inv_sqrt_dim_per_head = 1.0 / math.sqrt(dim_per_head)
        grid = lambda META: (
            min(
                triton.cdiv(B, META["YBLOCK"]) * triton.cdiv(T, META["XBLOCK"]),
                get_total_core_num(),
            ),
        )
        transform_bias_rescale_qkv_no_split_lowest_dim_impl[grid](
            qkv,
            qkv_bias,
            out_q,
            out_k,
            out_v,
            inv_sqrt_dim_per_head,
            B,
            T,
            D,
            num_heads,
            dim_per_head,
        )
    return out_q, out_k, out_v
