from .fsdp._init_utils import apply_init_utils_patch
from ._shard.api import apply_shard_tensor_patch
from ._shard.tensor_ops import apply_tensor_ops_patch
from .tensor._data_parallel_utils import apply_data_parallel_utils_patch
from .algorithms.default_hook import apply_default_hook_patch

def apply_distributed_patch():
    apply_init_utils_patch()
    apply_shard_tensor_patch()
    apply_tensor_ops_patch()
    apply_data_parallel_utils_patch()
    apply_default_hook_patch()
