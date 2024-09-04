from typing import Any, List, Optional, Tuple

import torch
from torch._inductor import ir
from torch._inductor import config as inductor_config
from torch._inductor.ir import FixedLayout, TensorBox
from torch._inductor.lowering import empty_strided, lowerings, fallback_handler
from torch._inductor.select_algorithm import autotune_select_algorithm, TritonTemplate
from torch._inductor.utils import is_dynamic
from torch._inductor.virtualized import V
from torch_mlu._inductor.lowering import mlu_register_lowering

aten = torch.ops.aten

NHWC_STRIDE_ORDER = [3, 0, 2, 1]
NCHW_STRIDE_ORDER = [3, 2, 1, 0]


# BLOCK_SIZE, num_warps, num_stage
def group_norm_nchw_default_config() -> Tuple[int, int, int]:
    return (1024, 1, 3)


# GROUP_SIZE, ITER_NUM, HW_N, XBLOCK, num_warps, num_stages
def group_norm_nhwc_config(
    n, h, w, channels, group_nums, max_autotune
) -> List[Tuple[int, int, int]]:
    group_size = channels // group_nums

    def get_iter_num(group_nums, group_size):
        for i in range(1, group_nums + 1):
            if group_nums % i == 0:
                if i * group_size >= 512:
                    return i
        return group_nums

    def get_hw_n(n, h, w, iter_num):
        hw = h * w
        for i in range(1, hw + 1):
            if hw % i == 0:
                if i * n * (group_nums // iter_num) >= 32:
                    return i
        return 1

    iter_num = get_iter_num(group_nums, group_size)
    hw_n = get_hw_n(n, h, w, iter_num)
    configs = [
        (group_size, iter_num, hw_n, 4, 1, 3),
    ]
    if max_autotune:
        configs += [
            (group_size, iter_num, hw_n, 8, 1, 3),
            (group_size, iter_num, hw_n, 16, 1, 3),
            (group_size, iter_num, hw_n, 32, 1, 3),
            (group_size, iter_num, hw_n, 64, 1, 3),
        ]
    return configs


def group_norm_nchw_grid(batch_groups, meta):
    return (batch_groups, 1, 1)


def group_norm_nhwc_grid(batch_groups, meta):
    batch_size = batch_groups // (meta["GROUP_NUM"])
    return (batch_size * (meta["GROUP_NUM"] // meta["ITER_NUM"]), meta["HW_N"], 1)


def group_norm_nhwc_tail_grid(n, c, h, w, meta):
    batch_size = n
    return (batch_size * (meta["GROUP_NUM"] // meta["ITER_NUM"]), meta["HW_N"], 1)


def is_nchw(stride):
    return False
    # stride_order = ir.get_stride_order(
    #     V.graph.sizevars.size_hints(stride)
    # )
    # return stride_order == NCHW_STRIDE_ORDER


def is_nhwc(stride):
    stride_order = ir.get_stride_order(V.graph.sizevars.size_hints(stride))
    return stride_order == NHWC_STRIDE_ORDER


group_norm_nhwc_template = TritonTemplate(
    name="group_norm_nhwc",
    grid=group_norm_nhwc_grid,
    source=r"""
{{def_kernel("IN", "MEAN", "VAR")}}
    pid = tl.program_id(0)
    pid_hw_id = tl.program_id(1)
    N = {{size("IN", 0)}}
    C = {{size("IN", 1)}}
    H = {{size("IN", 2)}}
    W = {{size("IN", 3)}}
    HW = H * W
    rnumel = HW * GROUP_SIZE
    rnumel_div = 1 / rnumel
    HW_SPLIT = HW // HW_N

    start = pid_hw_id * HW_SPLIT * C
    split_num = GROUP_NUM // ITER_NUM
    n = pid // split_num
    g = pid % split_num

    ws_offset = tl.arange(0, ITER_NUM)
    ws_mean_ptr = MEAN + n * GROUP_NUM + g * ITER_NUM + ws_offset
    ws_var_ptr = VAR + n * GROUP_NUM + g * ITER_NUM + ws_offset

    offset = start + n * HW * C + g * GROUP_SIZE * ITER_NUM
    xcnt = tl.cdiv(HW_SPLIT, XBLOCK)

    # truth
    block_ptr_in = tl.make_block_ptr(IN + offset, shape=[HW_SPLIT, ITER_NUM*GROUP_SIZE], strides=[C, 1],
                                     block_shape=[XBLOCK, ITER_NUM*GROUP_SIZE], order=[1, 0], offsets=[0, 0])
    mean = tl.zeros([ITER_NUM, GROUP_SIZE], tl.float32)
    var = tl.zeros([ITER_NUM, GROUP_SIZE], tl.float32)

    for idx in range(0, xcnt):
        tmp = tl.load(block_ptr_in, boundary_check=[0], padding_option='zero', eviction_policy='evict_last', cache_modifier=".cg")
        block_ptr_in = tl.advance(block_ptr_in, [XBLOCK, 0])
        tmp = tmp.reshape([XBLOCK, ITER_NUM, GROUP_SIZE])
        mean += tl.sum(tmp, axis=0, keep_dims=False)
        var += tl.sum(tmp*tmp, axis=0, keep_dims=False)

    mean = tl.sum(mean, axis=1, keep_dims=False)
    tl.atomic_add(ws_mean_ptr, mean)

    var = tl.sum(var, axis=1, keep_dims=False)

    var_offset = n * GROUP_NUM + g * ITER_NUM + ws_offset
    mask = var_offset < N * GROUP_NUM
    {{store_output(("var_offset",), "var", "mask")}}

    tl.atomic_add(ws_var_ptr, var)
""",
)

group_norm_nhwc_tail_template = TritonTemplate(
    name="group_norm_nhwc_tail",
    grid=group_norm_nhwc_grid,
    source=r"""
{{def_kernel("IN", "MEAN", "VAR", "TAIL_OUT", "WEIGHT", "BIAS")}}
    pid = tl.program_id(0)
    pid_hw_id = tl.program_id(1)
    N = {{size("IN", 0)}}
    C = {{size("IN", 1)}}
    H = {{size("IN", 2)}}
    W = {{size("IN", 3)}}
    HW = H * W
    rnumel = HW * GROUP_SIZE
    HW_SPLIT = HW // HW_N

    start = pid_hw_id * HW_SPLIT * C
    split_num = GROUP_NUM // ITER_NUM
    n = pid // split_num
    g = pid % split_num

    ws_offset = tl.arange(0, ITER_NUM)
    ws_mean_ptr = MEAN + n * GROUP_NUM + g * ITER_NUM + ws_offset
    ws_var_ptr = VAR + n * GROUP_NUM + g * ITER_NUM + ws_offset

    wb_offset = tl.arange(0, ITER_NUM * GROUP_SIZE)[None,:]
    weight_ptr = WEIGHT + g * ITER_NUM * GROUP_SIZE + wb_offset
    weight = tl.load(weight_ptr, cache_modifier=".cg").to(tl.float32)
    bias_ptr = BIAS + g * ITER_NUM * GROUP_SIZE + wb_offset
    bias = tl.load(bias_ptr, cache_modifier=".cg").to(tl.float32)

    offset = start + n * HW * C + g * GROUP_SIZE * ITER_NUM
    xcnt = tl.cdiv(HW_SPLIT, XBLOCK)

    # truth
    rnumel_div = 1 / rnumel
    mean = tl.load(ws_mean_ptr, cache_modifier=".cg")
    mean = mean * rnumel_div
    var = tl.load(ws_var_ptr, cache_modifier=".cg")
    var = var * rnumel_div - (mean * mean)
    var_norm = 1 / tl.sqrt(var + EPS)

    mean = tl.reshape(mean, [1, ITER_NUM])
    mean = tl.broadcast_to(mean, [GROUP_SIZE, ITER_NUM])
    mean = tl.trans(mean, 1, 0)
    mean = tl.reshape(mean, [1, ITER_NUM*GROUP_SIZE])
    mean = tl.broadcast_to(mean, [XBLOCK, ITER_NUM*GROUP_SIZE])

    var_div = tl.reshape(var_norm, [1, ITER_NUM])
    var_div = tl.broadcast_to(var_div, [GROUP_SIZE, ITER_NUM])
    var_div = tl.trans(var_div, 1, 0)
    var_div = tl.reshape(var_div, [1, ITER_NUM*GROUP_SIZE])
    var_div = tl.broadcast_to(var_div, [XBLOCK, ITER_NUM*GROUP_SIZE])

    weight = tl.broadcast_to(weight, [XBLOCK, ITER_NUM*GROUP_SIZE])
    var_div = var_div * weight
    bias = tl.broadcast_to(bias, [XBLOCK, ITER_NUM*GROUP_SIZE])

    block_ptr_in = tl.make_block_ptr(IN + offset, shape=[HW_SPLIT, ITER_NUM*GROUP_SIZE], strides=[C, 1],
                                     block_shape=[XBLOCK, ITER_NUM*GROUP_SIZE], order=[1, 0], offsets=[0, 0])
    block_ptr_out = tl.make_block_ptr(TAIL_OUT + offset, shape=[HW_SPLIT, ITER_NUM*GROUP_SIZE], strides=[C, 1],
                                      block_shape=[XBLOCK, ITER_NUM*GROUP_SIZE], order=[1, 0], offsets=[0, 0])
    dtype = TAIL_OUT.dtype.element_ty
    for idx in range(0, xcnt):
        tmp_in = tl.load(block_ptr_in, boundary_check=[0], padding_option='zero', eviction_policy='evict_last', cache_modifier=".cg").to(tl.float32)
        block_ptr_in = tl.advance(block_ptr_in, [XBLOCK, 0])
        tmp = tmp_in - mean
        tmp = tmp * var_div + bias
        tl.store(block_ptr_out, tmp.to(dtype), boundary_check=[0])
        block_ptr_out = tl.advance(block_ptr_out, [XBLOCK, 0])

    output_offset = n * GROUP_NUM + g * ITER_NUM + ws_offset
    mask = output_offset < N * GROUP_NUM
    {{store_output(("output_offset",), "var_norm", "mask")}}
""",
)

group_norm_nchw_template = TritonTemplate(
    name="group_norm_nchw",
    grid=group_norm_nchw_grid,
    source=r"""
{{def_kernel("IN", "OUT")}}
    pid = tl.program_id(0)
    in_ptr = IN + pid * RNUMEL
    out_ptr = OUT + pid * RNUMEL
    mean = tl.zeros([BLOCK_SIZE], tl.float32)
    var = tl.zeros([BLOCK_SIZE], tl.float32)

    block_ptr_in = tl.make_block_ptr(in_ptr, shape=[RNUMEL], strides=[1], block_shape=[BLOCK_SIZE], order=[0], offsets=[0])
    for idx in range(0, RNUMEL, BLOCK_SIZE):
        tmp = tl.load(block_ptr_in, boundary_check=[0], padding_option='zero', eviction_policy='evict_last', cache_modifier=".cg")
        block_ptr_in = tl.advance(block_ptr_in, [BLOCK_SIZE])
        mean += tmp
        var += tmp * tmp

    mean = tl.sum(mean) / RNUMEL
    var = tl.sum(var)  / RNUMEL - (mean * mean)
    var_div = 1.0 / tl.sqrt(var + EPS)

    dtype = OUT.dtype.element_ty
    block_ptr_in = tl.make_block_ptr(in_ptr, shape=[RNUMEL], strides=[1], block_shape=[BLOCK_SIZE], order=[0], offsets=[0])
    block_ptr_out = tl.make_block_ptr(out_ptr, shape=[RNUMEL], strides=[1], block_shape=[BLOCK_SIZE], order=[0], offsets=[0])
    for idx in range(0, RNUMEL, BLOCK_SIZE):
        tmp = tl.load(block_ptr_in, boundary_check=[0], padding_option='zero', eviction_policy='evict_first', cache_modifier=".cg").to(tl.float32)
        block_ptr_in = tl.advance(block_ptr_in, [BLOCK_SIZE])
        tmp = (tmp - mean) * var_div
        tl.store(block_ptr_out, tl.broadcast_to(tmp, [BLOCK_SIZE]).to(dtype), boundary_check=[0])
        block_ptr_out = tl.advance(block_ptr_out, [BLOCK_SIZE])

    mask = tl.full((1,), True, dtype=tl.int1)
    {{store_output(("pid", ), "var_div", "mask")}}
""",
)


def should_use_template(input, weight, bias):
    is_mean_rstd_used = len(V.current_node.users) > 1
    # The template implementation does not support outputting mean and rstd.
    if is_mean_rstd_used:
        return False
    input_size = input.get_size()
    if len(input_size) != 4:
        return False
    # The template implementation does not support dynamic shape.
    if is_dynamic(input):
        return False

    h = input_size[-2]
    c = input_size[1]
    # In some cases, the performance of templates is not as good as CNNL.
    if h < 32 or h // c > 2:
        return False
    if weight is None or bias is None:
        return False
    return True


def create_fallback_kernel(
    input: TensorBox,
    weight: Optional[TensorBox],
    bias: Optional[TensorBox],
    batch_size: int,
    num_channels: int,
    flattened_inner_size: int,
    num_groups: int,
    eps: float,
):
    if input is not None:
        input.realize()
    if weight is not None:
        weight.realize()
    if bias is not None:
        bias.realize()
    return fallback_handler(aten.native_group_norm.default)(
        input,
        weight,
        bias,
        batch_size,
        num_channels,
        flattened_inner_size,
        num_groups,
        eps,
    )


@mlu_register_lowering(aten.native_group_norm, type_promotion_kind=None)
def tuned_native_group_norm(
    input: TensorBox,
    weight: Optional[TensorBox],
    bias: Optional[TensorBox],
    batch_size: int,
    num_channels: int,
    flattened_inner_size: int,
    num_groups: int,
    eps: float,
):
    if not should_use_template(input, weight, bias):
        return create_fallback_kernel(
            input,
            weight,
            bias,
            batch_size,
            num_channels,
            flattened_inner_size,
            num_groups,
            eps,
        )

    if input is not None:
        input.realize()

    input_stride = V.current_node.all_input_nodes[0].meta["val"].stride()
    req_stride_order = ir.get_stride_order(V.graph.sizevars.size_hints(input_stride))
    input = ir.ExternKernel.require_stride_order(input, req_stride_order)
    if is_nchw(input_stride):
        choices: List[Any] = []
        configs: List[Tuple[int, int, int, int]] = []
        configs.append(group_norm_nchw_default_config())
        if inductor_config.max_autotune:
            configs += [
                (2048, 1, 3),
                (4096, 1, 3),
                (8192, 1, 3),
            ]

        norm_out = empty_strided(
            input.get_size(),
            input.get_stride(),
            dtype=input.get_dtype(),
            device=input.get_device(),
        )

        mean_shape = [
            batch_size * num_groups,
        ]
        layout = FixedLayout(
            input.get_device(),
            torch.float32,
            mean_shape,
            None,
        )

        xnumel = batch_size * num_groups
        h = input.get_size()[-2]
        w = input.get_size()[-1]
        rnumel = (num_channels // num_groups) * h * w
        for BLOCK_SIZE, num_warps, num_stages in configs:
            group_norm_nchw_template.maybe_append_choice(
                choices=choices,
                input_nodes=[input, norm_out],
                layout=layout,
                mutated_inputs=[
                    norm_out,
                ],
                num_stages=num_stages,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
                RNUMEL=rnumel,
                OUT_SIZE=xnumel,
                EPS=eps,
            )
        tmp_mean = autotune_select_algorithm(
            "group_norm_nchw", choices, [input, norm_out], layout
        )

        if weight is not None:
            weight_reshape = lowerings[aten.reshape](weight, (1, num_channels, 1, 1))
            norm_out = lowerings[aten.mul](norm_out, weight_reshape)
        if bias is not None:
            bias_reshape = lowerings[aten.reshape](bias, (1, num_channels, 1, 1))
            norm_out = lowerings[aten.add](norm_out, bias_reshape)
        return (norm_out,)

    elif is_nhwc(input_stride):
        mean_shape = [
            batch_size * num_groups,
        ]
        mean = fallback_handler(aten.full.default)(
            mean_shape, 0, dtype=torch.float32, device=input.get_device()
        )
        var = fallback_handler(aten.full.default)(
            mean_shape, 0, dtype=torch.float32, device=input.get_device()
        )

        layout = FixedLayout(
            input.get_device(),
            torch.float32,
            [
                batch_size * num_groups,
            ],
            None,
        )

        tail_out = empty_strided(
            input.get_size(),
            input.get_stride(),
            dtype=input.get_dtype(),
            device=input.get_device(),
        )

        h = input.get_size()[-2]
        w = input.get_size()[-1]
        choices: List[Any] = []
        configs = group_norm_nhwc_config(
            batch_size, h, w, num_channels, num_groups, inductor_config.max_autotune
        )
        for GROUP_SIZE, ITER_NUM, HW_N, XBLOCK, num_warps, num_stages in configs:
            group_norm_nhwc_template.maybe_append_choice(
                choices=choices,
                input_nodes=[input, mean, var],
                layout=layout,
                mutated_inputs=[mean, var],
                num_stages=num_stages,
                num_warps=num_warps,
                XBLOCK=XBLOCK,
                GROUP_SIZE=GROUP_SIZE,
                ITER_NUM=ITER_NUM,
                HW_N=HW_N,
                GROUP_NUM=num_groups,
            )
        new_var = autotune_select_algorithm(
            "group_norm_nhwc", choices, [input, mean, var], layout
        )

        if weight is not None:
            weight.realize()
        if bias is not None:
            bias.realize()

        input_nodes = [input, mean, var, tail_out, weight, bias]
        choices.clear()
        for GROUP_SIZE, ITER_NUM, HW_N, XBLOCK, num_warps, num_stages in configs:
            group_norm_nhwc_tail_template.maybe_append_choice(
                choices=choices,
                input_nodes=input_nodes,
                layout=layout,
                mutated_inputs=[tail_out],
                num_stages=num_stages,
                num_warps=num_warps,
                XBLOCK=XBLOCK,
                GROUP_SIZE=GROUP_SIZE,
                ITER_NUM=ITER_NUM,
                HW_N=HW_N,
                GROUP_NUM=num_groups,
                EPS=eps,
            )
        var_norm = autotune_select_algorithm(
            "group_norm_nhwc_tail", choices, input_nodes, layout
        )
        return (tail_out,)
    else:
        return create_fallback_kernel(
            input,
            weight,
            bias,
            batch_size,
            num_channels,
            flattened_inner_size,
            num_groups,
            eps,
        )
