import functools
from typing import cast, List, Tuple

import sympy

import torch
from torch._inductor.kernel.mm_common import acc_type, triton_config
from torch._inductor.utils import next_power_of_2
from torch._inductor.virtualized import V

from ...mlu._utils import replace_references


def mm_options(config, sym_m, sym_n, sym_k, layout, b_prologue_cast_type=None):
    """
    Common options to matmul triton templates.
    """
    even_k_symbolic = (
        # it isn't worth guarding on this
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"])
        == config.kwargs["BLOCK_K"]
    )
    # allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
    #     not inductor_config.force_same_precision
    #     or ((sym_m % 16) == 0 and (sym_n % 16) == 0 and (sym_k % 8) == 0)
    # )
    return dict(
        GROUP_M=8,
        EVEN_K=even_k_symbolic,
        ALLOW_TF32=False,
        ACC_TYPE=acc_type(layout.dtype),
        B_PROLOGUE_CAST_TYPE=b_prologue_cast_type,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        **config.kwargs,
    )


replace_references(torch._inductor.kernel.mm_common.mm_options, mm_options)


def filtered_configs(
    m: int,
    n: int,
    k: int,
    configs: List[Tuple[int, int, int, int, int]],
    has_int8_tensor=False,
):
    """Heuristic to shrink configs when they are bigger than the input size"""

    # According to https://github.com/openai/triton/issues/2156#issuecomment-1695897424
    # it's safer to use at least [32, 32] block size for int8/uint8
    # tensors
    # min_block_size = 32 if has_int8_tensor else 16
    min_block_size = 16
    m = max(next_power_of_2(V.graph.sizevars.size_hint(m)), min_block_size)
    n = max(next_power_of_2(V.graph.sizevars.size_hint(n)), min_block_size)
    k = max(next_power_of_2(V.graph.sizevars.size_hint(k)), min_block_size)
    used = set()
    for block_m, block_n, block_k, num_stages, num_warps in configs:
        # shrink configs for small sizes
        block_m = max(min(block_m, m), 16)
        block_n = max(min(block_n, n), min_block_size)
        block_k = max(min(block_k, k), min_block_size)
        # each warp computes 16x16 tile = 256
        num_warps = min(num_warps, block_m * block_n // 256)
        # NOTE(genesis): num_warps only support 1/4/8/16/32
        num_warps = next_power_of_2(num_warps)
        if num_warps == 2:
            num_warps = 4
        if (block_m, block_n, block_k, num_stages, num_warps) not in used:
            used.add((block_m, block_n, block_k, num_stages, num_warps))
            yield triton_config(
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_stages=num_stages,
                num_warps=num_warps,
            )


# from genesis/Python/triton/ops/matmul.py
def get_configs_io_bound():
    configs = []
    # for num_stages in [2, 3, 4, 5, 6]:
    for num_stages in [1]:
        # for block_m in [16, 32]:
        for block_m in [32]:
            # for block_k in [32, 64]:
            for block_k in [64]:
                for block_n in [32, 64, 128, 256]:
                    # num_warps = 1 if block_n <= 64 else 4
                    num_warps = 1
                    configs.append(
                        {
                            "config": (
                                block_m,
                                block_n,
                                block_k,
                                num_stages,
                                num_warps,
                            ),
                            "cond": True,
                        }
                    )
    return configs


# List of dictionaries to store the kernel configs. Configs that evaluate to true
# will be utilised on the target platform
mm_kernel_configs = [
    # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
    # {"config": (32, 32, 16, 1, 1), "cond": True},
    # {"config": (32, 32, 32, 1, 1), "cond": True},
    # {"config": (64, 64, 64, 1, 1), "cond": True},
    {"config": (128, 64, 64, 1, 1), "cond": True},
    {"config": (256, 64, 64, 1, 1), "cond": True},
    {"config": (512, 64, 128, 1, 1), "cond": True},
    {"config": (1024, 64, 128, 1, 1), "cond": True},
    # {"config": (1024, 64, 256, 1, 1), "cond": True},
    # {"config": (1024, 128, 128, 1, 1), "cond": True},
    # {"config": (1024, 256, 128, 1, 1), "cond": True},
]

# mm_kernel_configs = get_configs_io_bound()

mm_platform_configs = tuple(
    cast(Tuple[int, int, int, int, int], config["config"])
    for config in mm_kernel_configs
    if config["cond"]
)

mm_configs = functools.partial(
    filtered_configs,
    configs=mm_platform_configs,
)

replace_references(torch._inductor.kernel.mm_common.mm_configs, mm_configs)
