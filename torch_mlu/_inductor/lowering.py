import logging

import torch
from torch._inductor.lowering import (
    register_lowering,
    lowerings,
    add_needs_realized_inputs,
    add_layout_constraint,
    fallback_handler,
    FALLBACK_ALLOW_LIST,
)
from torch._inductor import inductor_prims
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._inductor.decomposition import decompositions

from ..mlu._utils import replace_references

log = logging.getLogger(__name__)

# fix cudagraph tests
FALLBACK_ALLOW_LIST.add("aten::clone")


def make_fallback(op, layout_constraint=None, warn=True):
    assert op not in decompositions, f"both a fallback and a decomp for same op: {op}"
    if 0:
        # Note: 'warn' is holdover from when this was a warning, but for ops that previously
        # set warn=False we do not want a CI error.
        # Ignore the 'suppress errors' configs in CI, as this particular warning happens on startup anyway and is not
        # likely to be triggered preferentially on one CI config over another.
        if torch._dynamo.config.suppress_errors:
            torch._dynamo.config.suppress_errors = False
            log.warning(
                "A make_fallback error occurred in suppress_errors config,"
                " and suppress_errors is being disabled to surface it."
            )
        raise AssertionError(
            f"make_fallback({op}): a decomposition exists, we should switch to it."
            " To fix this error, either add a decomposition to core_aten_decompositions (preferred)"
            " or inductor_decompositions, and delete the corresponding `make_fallback` line."
            " Get help from the inductor team if unsure, don't pick arbitrarily to unblock yourself.",
        )

    def register_fallback(op_overload):
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    if isinstance(op, torch._ops.OpOverloadPacket):
        for ol in op.overloads():
            op_overload = getattr(op, ol)
            register_fallback(op_overload)
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        register_fallback(op)
    else:
        raise RuntimeError(f"Unsupported fallback {op} with type {type(op)}")


replace_references(torch._inductor.lowering.make_fallback, make_fallback)


def delete_lowerings(aten_fn):
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn in lowerings:
                    lowerings.pop(other_fn)
        elif isinstance(fn, torch._ops.OpOverload):
            if fn in lowerings:
                lowerings.pop(fn, None)


def mlu_register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
):
    delete_lowerings(aten_fn)
    return register_lowering(
        aten_fn, broadcast, type_promotion_kind, convert_input_to_bool
    )


def remove_register_lowering(op_list=[]):
    for op in op_list:
        delete_lowerings(op)


aten = torch.ops.aten
prims = torch.ops.prims

# Exclude below lowerings to get a better performance for now.
# The following blacklist will be gradually reduced to incorporate more operators.
remove_list = [
    # aten.mean,
    # aten.var,
    # aten.var_mean,
    aten._convolution,
    aten.convolution_backward,
    aten._upsample_bicubic2d_aa,
    aten.addmm,
    aten.bmm,
    aten.cat,
    aten.clone,
    aten.convolution,
    aten.diagonal,
    aten.diagonal_scatter,
    aten.embedding,
    aten.expand,
    aten.gather,
    aten.index_put,
    aten.index_put_,
    aten.max_pool2d_with_indices,
    aten.max_pool2d_with_indices_backward,
    aten.mm,
    aten.permute,
    aten.reshape,
    aten.scatter,
    aten.scatter_,
    aten.scatter_add,
    aten.scatter_add_,
    aten.scatter_reduce,
    aten.scatter_reduce_,
    aten.select_scatter,
    aten.slice,
    aten.slice_scatter,
    aten.split,
    aten.squeeze,
    aten.squeeze_,
    aten.unfold,
    aten.upsample_bicubic2d,
    aten.upsample_bilinear2d,
    aten.upsample_nearest2d,
    aten.view,
    aten.where,
]
remove_register_lowering(remove_list)
mlu_lowerings = list(lowerings.keys())
