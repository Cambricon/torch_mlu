from typing import Callable, Iterable, Optional, Tuple, Type, Union

import torch
from torch._dynamo.device_interface import (
    register_interface_for_device,
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
    _device_t,
    device_interfaces,
)

from ..mlu._utils import replace_references

get_mlu_stream: Optional[Callable[[int], int]]
if torch.mlu._is_compiled():
    from torch_mlu._MLUC import _mlu_getCurrentMLUStream as get_mlu_stream
else:
    get_mlu_stream = None

_device_initialized = False


class MluInterface(DeviceInterface):
    device = torch.mlu.device

    # register Event and Stream class into the backend interface
    # make sure Event and Stream are implemented and inherited from the _EventBase and _StreamBase
    Event = torch.mlu.Event
    Stream = torch.mlu.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["mlu"] = device

        @staticmethod
        def current_device() -> int:
            if "mlu" in caching_worker_current_devices:
                return caching_worker_current_devices["mlu"]
            return torch.mlu.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "mlu"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = MluInterface.Worker.current_device()

            if "mlu" not in caching_worker_device_properties:
                device_prop = [
                    torch.mlu.get_device_properties(i)
                    for i in range(torch.mlu.device_count())
                ]
                caching_worker_device_properties["mlu"] = device_prop

            return caching_worker_device_properties["mlu"][device]

    current_device = staticmethod(torch.mlu.current_device)
    set_device = staticmethod(torch.mlu.set_device)
    device_count = staticmethod(torch.mlu.device_count)
    stream = staticmethod(torch.mlu.stream)  # type: ignore[assignment]
    current_stream = staticmethod(torch.mlu.current_stream)
    set_stream = staticmethod(torch.mlu.set_stream)  # type: ignore[assignment]
    # _set_stream_by_id = staticmethod(torch.mlu._set_stream_by_id)  # type: ignore[assignment]
    synchronize = staticmethod(torch.mlu.synchronize)
    get_device_properties = staticmethod(torch.mlu.get_device_properties)  # type: ignore[assignment]
    get_raw_stream = staticmethod(get_mlu_stream)  # type: ignore[arg-type]

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.mlu.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        # For MLU (Machine Learning Units), isa_version represents a more detailed
        # and accurate target for compilation.
        return torch.mlu.get_device_properties(device).isa_version


def get_interface_for_device(device: Union[str, torch.device]) -> Type[DeviceInterface]:
    if isinstance(device, torch.device):
        device = str(device)
    if not _device_initialized:
        init_device_reg()
    if device in device_interfaces:
        return device_interfaces[device]
    raise NotImplementedError(f"No interface for device {device}")


replace_references(
    torch._dynamo.device_interface.get_interface_for_device, get_interface_for_device
)


def get_registered_device_interfaces() -> Iterable[Tuple[str, Type[DeviceInterface]]]:
    if not _device_initialized:
        init_device_reg()
    return device_interfaces.items()


replace_references(
    torch._dynamo.device_interface.get_registered_device_interfaces,
    get_registered_device_interfaces,
)


def init_device_reg():
    global _device_initialized
    register_interface_for_device("mlu", MluInterface)
    for i in range(torch.mlu.device_count()):
        register_interface_for_device(f"mlu:{i}", MluInterface)
    _device_initialized = True

    # enable linear memory for MLU
    torch.mlu.memory.enable_linear_memory()
