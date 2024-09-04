import sys
from typing import Union
import torch
from torch.nn.parallel.scatter_gather import (  # type: ignore[attr-defined]
    _is_namedtuple,
)
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.distributed_c10d import (
    _get_default_group,
    get_backend,
    _GLOO_AVAILABLE,
    _check_p2p_op_list,
    _coalescing_manager,
)
from torch._C._distributed_c10d import _ProcessGroupWrapper
import torch_mlu


_CNCL_AVAILABLE = True


def version():
    major, minor, patch = torch_mlu._MLUC._cncl_version()
    return (major, minor, patch)


def is_cncl_available():
    return _CNCL_AVAILABLE


torch.distributed.__setattr__("is_cncl_available", is_cncl_available)


def batch_isend_irecv(p2p_op_list):
    """
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in ``p2p_op_list`` and return the corresponding
    requests. CNCL, Gloo, and UCC backend are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``torch.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed request objects returned by calling the corresponding
        op in the op_list.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> send_tensor = torch.arange(2) + 2 * rank
        >>> recv_tensor = torch.randn(2)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
        >>> recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1

    .. note:: Note that when this API is used with the CNCL PG backend, users must set
        the current GPU device with `torch.cuda.set_device`, otherwise it will
        lead to unexpected hang issues.

        In addition, if this API is the first collective call in the ``group``
        passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        this API call; otherwise, the behavior is undefined. If this API call is
        not the first collective call in the ``group``, batched P2P operations
        involving only a subset of ranks of the ``group`` are allowed.
    """
    _check_p2p_op_list(p2p_op_list)
    group = p2p_op_list[0].group
    device = p2p_op_list[0].tensor.device
    if device.type == "mlu":
        # CNCL style coalescing
        with _coalescing_manager(group, device, async_ops=True) as cm:
            for p2p_op in p2p_op_list:
                p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        return cm.works
    else:
        # Backward support for Gloo
        reqs = []
        for p2p_op in p2p_op_list:
            work = p2p_op.op(p2p_op.tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
            if work:
                reqs.append(work)
        return reqs


torch.distributed.__setattr__("batch_isend_irecv", batch_isend_irecv)

if not hasattr(torch.distributed.Backend, "CNCL"):
    torch.distributed.Backend.register_backend(
        "cncl",
        lambda store, group_rank, group_size, timeout: torch_mlu._MLUC.ProcessGroupCNCL(
            store, group_rank, group_size, timeout
        ),
        devices=["mlu"],
    )

if hasattr(torch_mlu._MLUC, "ProcessGroupCNCL"):
    from torch_mlu._MLUC import ProcessGroupCNCL

    torch.distributed.__setattr__("ProcessGroupCNCL", ProcessGroupCNCL)
    ProcessGroupCNCL.__module__ = "torch.distributed.distributed_c10d"
