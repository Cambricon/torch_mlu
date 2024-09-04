import torch
import torch_mlu

__all__ = ["amp_definitely_not_available"]


def amp_definitely_not_available():
    return not torch.mlu.is_available()
