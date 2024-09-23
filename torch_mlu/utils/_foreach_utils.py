from typing import List
import torch


def _get_foreach_kernels_supported_devices() -> List[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu"]


def apply_foreach_fused_patch():
    torch.utils._foreach_utils._get_foreach_kernels_supported_devices.__code__ = (
        _get_foreach_kernels_supported_devices.__code__
    )
