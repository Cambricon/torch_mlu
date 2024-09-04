import copy
import torch
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    ShardedTensor,
)


@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    # Validate types
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    dev: torch.device
    if self_st._local_shards:
        dev = self_st._local_shards[0].tensor.device
    elif pg and pg._get_backend_name() == "gloo":
        dev = torch.device("cpu")
    else:
        dev = torch.device(torch.mlu.current_device())
    return dev


def apply_tensor_ops_patch():
    torch.distributed._shard.sharded_tensor._ops.tensor_ops.tensor_device = (
        tensor_device
    )
