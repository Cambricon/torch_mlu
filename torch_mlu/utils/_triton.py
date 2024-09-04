import functools
import hashlib

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.utils._triton import has_triton_package


@functools.lru_cache(None)
def has_triton() -> bool:
    return has_triton_package()


torch.utils._triton.has_triton = has_triton


@functools.lru_cache(None)
def triton_backend():
    import torch

    if torch.version.hip:
        # Does not work with ROCm
        return None

    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    target = driver.active.get_current_target()
    return make_backend(target)


@functools.lru_cache(None)
def triton_hash_with_backend():
    import torch

    from triton.compiler.compiler import triton_key

    backend = triton_backend()
    key = f"{triton_key()}-{backend.hash()}"

    # Hash is upper case so that it can't contain any Python keywords.
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()


torch.utils._triton.triton_hash_with_backend = triton_hash_with_backend
