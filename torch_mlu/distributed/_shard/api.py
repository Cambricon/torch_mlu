import torch
import torch.distributed as dist


# Monkey-patch torch.distributed._shard.sharded_tensor.ShardedTensor._get_preferred_device,
# add CNCL backend
def _get_preferred_device(self) -> torch.device:
    """
    Return the preferred device to be used when creating tensors for collectives.
    This method takes into account the associated process group
    """
    if dist.get_backend(self._process_group) == dist.Backend.NCCL:
        return torch.device(torch.cuda.current_device())
    if dist.get_backend(self._process_group) == dist.Backend.CNCL:
        return torch.device(torch.mlu.current_device())
    return torch.device("cpu")


def apply_shard_tensor_patch():
    torch.distributed._shard.sharded_tensor.ShardedTensor._get_preferred_device = (
        _get_preferred_device
    )
