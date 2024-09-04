import torch
import functools

from torch.utils._triton import has_triton
from torch.testing._internal.inductor_utils import (
    HAS_CPU,
    skipCPUIf,
    skipDeviceIf,
    _check_has_dynamic_shape,
)

HAS_CUDA = torch.mlu.is_available() and has_triton()
HAS_GPU = HAS_CUDA

GPUS = ["mlu"]

HAS_MULTIGPU = any(
    getattr(torch, mlu).is_available() and getattr(torch, mlu).device_count() >= 2
    for mlu in GPUS
)

GPU_TYPE = "mlu"

skipCUDAIf = functools.partial(skipDeviceIf, device="mlu")

HAS_CPU = HAS_CPU
skipCPUIf = skipCPUIf
_check_has_dynamic_shape = _check_has_dynamic_shape
