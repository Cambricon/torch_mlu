import torch
from torch import Tensor


def active_sigmoid_mlu(x: Tensor):
    return torch.ops.mlu_custom_ext.active_sigmoid_mlu(x)
