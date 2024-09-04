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
    return torch.is_autocast_enabled(torch._C._get_privateuse1_backend_name())


def set_autocast_enabled(enable):
    return torch.set_autocast_enabled(torch._C._get_privateuse1_backend_name(), enable)


def get_autocast_dtype():
    return torch.get_autocast_dtype(torch._C._get_privateuse1_backend_name())


def set_autocast_dtype(dtype):
    return torch.set_autocast_dtype(torch._C._get_privateuse1_backend_name(), dtype)
