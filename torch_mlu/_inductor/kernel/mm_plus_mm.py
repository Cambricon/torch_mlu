import functools

import torch

aten = torch.ops.aten


@functools.lru_cache(None)
def mm_configs():
    import triton

    # List of dictionaries to store the kernel configs. Configs that evaluate to true
    # will be utilised on the target platform
    mm_triton_configs = [
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 3,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 16,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            "num_stages": 1,
            "num_warps": 8,
            "cond": torch.version.hip is None,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16},
            "num_stages": 1,
            "num_warps": 4,
            "cond": True,
        },
    ]

    # Filter out configs in which cond evaluates to true
    # On ROCm convert num_stages to 1 as pipelining provides no benefit
    if torch.version.hip:
        filtered_configs = [
            triton.Config(c["config"], num_stages=1, num_warps=c["num_warps"])
            for c in mm_triton_configs
            if c["cond"]
        ]
    else:
        filtered_configs = [
            triton.Config(
                c["config"], num_stages=c["num_stages"], num_warps=c["num_warps"]
            )
            for c in mm_triton_configs
            if c["cond"]
        ]

    return filtered_configs


torch._inductor.kernel.mm_plus_mm.mm_configs = mm_configs
