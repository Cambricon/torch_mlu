import torch
import torch_mlu
from typing import Optional
from typing import Any, List


class _LazySeedTracker:
    # Since seeding is memory-less, only track the latest seed.
    # Note: `manual_seed_all` followed by `manual_seed` overwrites
    # the seed on current device. We track the order of **latest**
    # calls between these two API.
    def __init__(self):
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = []

    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> List:
        return self.call_order


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a MLU device. Note that for a MLU device without a specified index,
    i.e., ``torch.device('mlu')``, this will return the current default MLU
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MLU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["mlu", "cpu"]:
                raise ValueError(
                    "Expected a mlu or cpu device, but got: {}".format(device)
                )
        elif device.type != "mlu":
            raise ValueError("Expected a mlu device, but got: {}".format(device))
    if not torch.jit.is_scripting():
        if isinstance(device, torch.mlu.device):
            return device.idx
    return torch._utils._get_device_index(device, optional, allow_cpu)


def _dummy_type(name: str) -> type:
    def get_err_fn(is_init: bool):
        def err_fn(obj, *args, **kwargs):
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )
