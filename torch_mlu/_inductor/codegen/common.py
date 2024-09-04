import torch
from torch._inductor.codegen.common import DeviceOpOverrides, device_op_overrides_dict
from ...mlu._utils import replace_references


def get_device_op_overrides(device: str):
    assert isinstance(device, str)

    if not device_op_overrides_dict.keys():
        # Modified by Cambricon start: replace with new codes
        from . import device_op_overrides  # noqa: F401

        # Original codes:
        # from .cuda import device_op_overrides  # noqa: F401
        # Modified by Cambricon end

    if device in device_op_overrides_dict.keys():
        return device_op_overrides_dict[device]

    return DeviceOpOverrides()


replace_references(
    torch._inductor.codegen.common.get_device_op_overrides, get_device_op_overrides
)
