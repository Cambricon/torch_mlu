from typing import Tuple
import sys
import torch
from torch.utils.checkpoint import DefaultDeviceType
from torch_mlu import _MLUC
import torch_mlu.mlu
import torch_mlu.mlu.amp
import torch_mlu.mlu.cnpx
import torch_mlu.profiler
import torch_mlu.optimizers
import torch_mlu.backends
from torch_mlu.backends.cnnl import CnnlModule
from torch_mlu.backends.mlufusion import MlufusionModule
from torch_mlu.utils.counter import _check_gencase
from torch_mlu.utils import apply_module_patch
from torch_mlu.data import apply_dataloader_patch, apply_pin_memory_patch
from torch_mlu.nn import apply_functional_patch
from torch_mlu.distributed import apply_distributed_patch
from torch_mlu.optimizers import apply_optim_patch
from torch_mlu.mlu import apply_storage_patch, apply_reductions_patch
_check_gencase()

def get_version():
    return _MLUC._get_version()

__version__ = get_version()

def get_git_version() -> str:
    try:
        from torch_mlu.version import git_version
        return git_version
    except Exception:
        pass
    return "unknown"

def apply_patches():
    apply_module_patch()
    apply_dataloader_patch()
    apply_pin_memory_patch()
    apply_functional_patch()
    apply_distributed_patch()
    apply_optim_patch()


apply_patches()


def _check_register_once(module, attr):
    if hasattr(module, attr):
        raise RuntimeError(f"The custom device module of {module} has already been registered with {attr}")

torch.utils.rename_privateuse1_backend("mlu")
torch._register_device_module('mlu', torch_mlu.mlu)

custom_backend_name = 'mlu'

# tensor.mlu
def wrapper_tensor_mlu(self, *args, **kwargs):
    return torch_mlu._MLUC.mlu(self, *args, **kwargs)
_check_register_once(torch.Tensor, custom_backend_name)
setattr(torch.Tensor, custom_backend_name, wrapper_tensor_mlu)

# tensor.is_mlu
@property
def wrapper_tensor_is_mlu(t: torch.Tensor) -> bool:
    return t.device.type == 'mlu'
_check_register_once(torch.Tensor, f'is_{custom_backend_name}')
setattr(torch.Tensor, f'is_{custom_backend_name}', wrapper_tensor_is_mlu)


unsupported_dtype = [
    torch.quint8, torch.quint4x2,
    torch.quint2x4, torch.qint32, torch.qint8
]

# tensor and storage have custom implementation, we generate methods in other way
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=False, for_module=True, for_storage=True,
    unsupported_dtype=unsupported_dtype)

### torch.backends
torch._C._set_cublas_allow_tf32 = torch_mlu.backends.mlu.fake_set_cublas_allow_tf32
torch._C._set_cublas_allow_fp16_reduced_precision_reduction = torch_mlu.backends.mlu.fake_set_cublas_allow_fp16_reduced_precision_reduction
torch.backends.__setattr__("mlu", torch_mlu.backends.mlu)
sys.modules["torch.backends.mlu"] = torch_mlu.backends.mlu
torch.backends.__setattr__("cnnl", CnnlModule(torch_mlu.backends.cnnl, "torch.backends.cnnl"))
torch.backends.__setattr__("mlufusion", MlufusionModule(torch_mlu.backends.mlufusion, "torch.backends.mlufusion"))

_MLUC._initExtension()

torch._C._conv_determine_backend_memory_format = torch_mlu._MLUC._conv_determine_backend_memory_format
### torch compile
from torch_mlu.utils import _triton
import torch_mlu._dynamo
import torch_mlu._inductor

### tf32 cnmatmul control
setattr(torch._C, "_get_cnmatmul_allow_tf32", torch_mlu._MLUC._get_cnmatmul_allow_tf32)
setattr(torch._C, "_set_cnmatmul_allow_tf32", torch_mlu._MLUC._set_cnmatmul_allow_tf32)
# set default device type mlu for checkpointing
DefaultDeviceType.set_device_type("mlu")

torch.version.mlu = None

apply_storage_patch()
apply_reductions_patch()
