import torch
import torch_mlu
from typing import Optional
from typing import Any
import gc


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a MLU device. Note that for a MLU device without a specified index,
    i.e., ``torch.device('mlu')``, this will return the current default MLU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MLU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["mlu", "cpu"]:
                raise ValueError(f"Expected a mlu or cpu device, but got: {device}")
        elif device.type != "mlu":
            raise ValueError(f"Expected a mlu device, but got: {device}")
    if not torch.jit.is_scripting():
        if isinstance(device, torch.mlu.device):
            return device.idx
    return torch._utils._get_device_index(device, optional, allow_cpu)


def replace_references(original_object, new_object):
    referrers = gc.get_referrers(original_object)

    for referrer in referrers:
        if isinstance(referrer, dict):
            for key, value in referrer.items():
                if value is original_object:
                    referrer[key] = new_object
        elif isinstance(referrer, list):
            for i, item in enumerate(referrer):
                if item is original_object:
                    referrer[i] = new_object
