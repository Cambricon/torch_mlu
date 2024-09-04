import torch
import torch_mlu


def get_current_device_index() -> int:
    r"""Checks if there are MLU devices available and
    returns the device index of the current default MLU device.
    Returns -1 in case there are no MLU devices available.
    Arguments: ``None``
    """
    if torch.mlu.device_count() > 0:
        return torch.mlu.current_device()
    return -1


def _get_available_device_type():
    if torch.mlu.is_available():
        return "mlu"
    return None


def _get_device_attr(get_member):
    device_type = _get_available_device_type()
    if device_type and device_type.lower() == "mlu":
        return get_member(torch.mlu)
    return None


# monkey patch three functions called by torch._utils._get_device_index
torch._utils.__setattr__("get_current_device_index", get_current_device_index)
torch._utils.__setattr__("_get_device_attr", _get_device_attr)
torch._utils.__setattr__("_get_available_device_type", _get_available_device_type)
