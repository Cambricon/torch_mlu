import torch
from torch import Tensor


def active_sigmoid_mlu(x: Tensor):
    return torch.ops.mlu_custom_ext.active_sigmoid_mlu(x)


@torch.library.impl_abstract("mlu_custom_ext::active_sigmoid_mlu")
def meta_active_sigmoid_mlu(x):
    return torch.empty_like(x)
