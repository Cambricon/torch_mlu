from .fsdp._init_utils import apply_init_utils_patch
from .fsdp._optim_utils import apply_optim_utils_patch
from ._shard.api import apply_shard_tensor_patch
from ._shard.tensor_ops import apply_tensor_ops_patch
from .tensor.fsdp import apply_fsdp_patch
from .algorithms.default_hook import apply_default_hook_patch
from .nn.functional import apply_functional_patch

def apply_distributed_patch():
    apply_init_utils_patch()
    apply_optim_utils_patch()
    apply_shard_tensor_patch()
    apply_tensor_ops_patch()
    apply_fsdp_patch()
    apply_default_hook_patch()
    apply_functional_patch()
