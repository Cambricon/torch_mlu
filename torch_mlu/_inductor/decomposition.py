from typing import Callable, Dict, Sequence, Union

import torch
from torch._ops import OpOverload, OpOverloadPacket

from torch._inductor.decomposition import decompositions, remove_decompositions
from torch._decomp.decompositions_for_jvp import decomposition_table_for_jvp

aten = torch.ops.aten
DispatchKey = torch._C.DispatchKey


def remove_py_kernels(aten_ops):
    dispatch_keys = [DispatchKey.Autograd, DispatchKey.CompositeImplicitAutograd]
    for op in aten_ops:
        if hasattr(op, "py_kernels"):
            for key in dispatch_keys:
                op.py_kernels.pop(key, None)


# Exclude below decompositions to get a better performance for now.
# The following blacklist will be gradually reduced to incorporate more operators.
mlu_python_override_to_exclude = [
    aten.native_batch_norm.default,
    aten.upsample_bicubic2d.vec,
    aten.upsample_bilinear2d.default,
    aten.upsample_bilinear2d.vec,
    aten.upsample_nearest2d.default,
]
remove_py_kernels(mlu_python_override_to_exclude)

mlu_decomps_to_exclude = [
    aten._log_softmax,
    aten._log_softmax_backward_data,
    aten._reshape_alias,
    aten._softmax,
    aten._softmax_backward_data,
    aten._unsafe_view,
    aten._upsample_bicubic2d_aa,
    aten.addmm,
    aten.bmm,
    aten.cat,
    aten.convolution_backward,
    aten.embedding,
    aten.embedding_dense_backward,
    aten.gelu,
    aten.mm,
    aten.native_batch_norm,
    aten.native_dropout,
    aten.native_group_norm,
    aten.native_layer_norm,
    aten.native_layer_norm_backward,
    aten.rand_like,
    aten.randn_like,
    aten.upsample_bicubic2d,
    aten.upsample_bilinear2d,
    aten.upsample_nearest2d,
]

mlu_jvp_decomps_to_exclude = [
    aten.native_layer_norm_backward,
]

remove_decompositions(decompositions, mlu_decomps_to_exclude)
remove_decompositions(decomposition_table_for_jvp, mlu_jvp_decomps_to_exclude)
