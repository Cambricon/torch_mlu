import torch
import torch_mlu

__all__ = [
    "get_amp_supported_dtype",
    "is_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_enabled",
    "set_autocast_dtype",
]


def get_amp_supported_dtype():
    return [torch.float16, torch.bfloat16]


def is_autocast_enabled():
    return torch_mlu._MLUC.is_autocast_enabled()


def get_autocast_dtype():
    return torch_mlu._MLUC.get_autocast_dtype()


def set_autocast_enabled(enable):
    torch_mlu._MLUC.set_autocast_enabled(enable)


def set_autocast_dtype(dtype):
    return torch_mlu._MLUC.set_autocast_dtype(dtype)
