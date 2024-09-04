import functools
from typing import Union, Tuple, Optional, TypeVar, Any

import torch
import torch_mlu
from ._utils import _get_device_index

if hasattr(torch_mlu._MLUC, "_mlu_exchangeDevice"):
    _exchange_device = torch_mlu._MLUC._mlu_exchangeDevice
else:

    def _exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without MLU support")


if hasattr(torch_mlu._MLUC, "_mlu_maybeExchangeDevice"):
    _maybe_exchange_device = torch_mlu._MLUC._mlu_maybeExchangeDevice
else:

    def _maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without MLU support")


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = _maybe_exchange_device(self.prev_idx)
        return False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = _maybe_exchange_device(self.prev_idx)
        return False


Device = device
_device_t = Union[device, str, int, None]


### Device management
def set_device(device: _device_t) -> None:
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``MLU_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device, optional=True)
    if device >= 0:
        torch_mlu._MLUC._mlu_setDevice(device)


def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    torch.mlu._lazy_init()
    return torch_mlu._MLUC._mlu_getDevice()


def get_device_properties(device: _device_t):
    r"""Gets the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _MLUDeviceProperties: the properties of the device
    """
    torch.mlu._lazy_init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= torch.mlu.device_count():
        raise AssertionError("Invalid device id")
    return torch_mlu._MLUC._get_device_properties(device)


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Gets the cuda capability of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.mlu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device. eg. MLU370
    """
    return get_device_properties(device).name


@functools.lru_cache(32)
def supports_linear_memory(device: Optional[_device_t] = None) -> bool:
    r"""Returns a boolean indicating if the device supports linear memory.
    Args:
        device (torch.device or int, optional): device for which to return the
            property. It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return get_device_properties(device).supports_linear_memory


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on a MLU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    torch.mlu._lazy_init()
    with Device(device):
        torch_mlu._MLUC._mlu_synchronize()


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use tensors as arguments. If a given object is
    not allocated on a MLU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.device.type == "mlu" else -1
        super().__init__(idx)


def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    r"""Checks if peer access between two devices is possible."""
    torch.mlu._lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= torch.mlu.device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= torch.mlu.device_count():
        raise AssertionError("Invalid peer device id")
    return torch_mlu._MLUC._mlu_canDeviceAccessPeer(device, peer_device)


__all__ = [
    "_DeviceGuard",
    "device",
    "set_device",
    "current_device",
    "get_device_properties",
    "get_device_capability",
    "get_device_name",
    "supports_linear_memory",
    "synchronize",
    "device_of",
    "can_device_access_peer",
]
