import sys
import os
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
    GroupMember,
    _update_default_pg,
    _world,
)
from torch._C._distributed_c10d import (
    _ProcessGroupWrapper,
    _unregister_process_group,
    _unregister_process_group,
    _unregister_all_process_groups,
)
from typing import Optional
import warnings

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


def _shutdown_backend(pg):
    """
    Try to shut down the backend of a process group.
    """
    backend = None
    try:
        backend = pg._get_backend(torch.device("mlu"))
    except RuntimeError:
        pass
    if is_cncl_available() and isinstance(backend, ProcessGroupCNCL):
        # explictly call shutdown to ensure that CNCL resources are released
        backend._shutdown()


def _abort_in_destroy_pg() -> bool:
    # Environment variable to control whether to abort the communicators when users call destroy_process_group()
    env = os.getenv("TORCH_CNCL_ABORT_IN_DESTROY_PG", "0")
    return env == "1" or env.lower() == "true"


def destroy_process_group(group: Optional[ProcessGroup] = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    global _world

    if group == GroupMember.NON_GROUP_MEMBER:
        return

    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group

    assert pg is not None
    if _world.pg_map.get(pg, None) is None:
        raise ValueError("Invalid process group specified")

    # When users register Python onCompletion hooks, those hooks will run on a
    # different thread than the main thread. Today, the ProcessGroup dtor does
    # wait for that thread. However, the dtor might finish after the Python
    # Interpreter exits. After that grabbing the GIL for the Python hook will crash.
    # We can either revive the interpreter when running hooks or keep the main one
    # alive until all works and hooks are done. The current implementation does the
    # latter. Therefore, we explicitly call _wait_for_pending_works() here to wait
    # for the pending hooks to finish.
    if pg.name().lower() == "cncl" and pg._has_hooks():
        pg._wait_for_pending_works()

    if group is None or group == GroupMember.WORLD:
        if _abort_in_destroy_pg():
            for pg_to_shutdown in sorted(
                _world.pg_names, key=lambda x: _world.pg_names[x], reverse=True
            ):
                _shutdown_backend(pg_to_shutdown)

        _update_default_pg(None)
        _world.pg_map.clear()
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_backend_config.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.pg_coalesce_state.clear()
        _world.pg_default_device.clear()
        _unregister_all_process_groups()

        # when process group doesn't have an explicit name (only WORLD (default)
        # process group can have an explicit name), we use global _world.group_count
        # to generate the name. We need to reset the counter on destruction to
        # allow consistent value to be generated when we re-create process
        # groups after some trainers recover from failure
        #
        # We only reset this when WORLD is being destroyed because if this
        # process group is in good state, we aren't dealing with failures.
        _world.group_count = 0
    else:
        if _abort_in_destroy_pg():
            _shutdown_backend(pg)
        del _world.pg_map[pg]
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        del _world.pg_backend_config[pg]
        if pg in _world.pg_default_device:
            del _world.pg_default_device[pg]
        if pg in _world.pg_coalesce_state.keys():
            warnings.warn(
                "Some coalesced collectives haven't been launched when "
                "ProcessGroup is destroyed. They will be cleaned."
            )
            del _world.pg_coalesce_state[pg]

        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass
        _unregister_process_group(pg.group_name)


torch.distributed.__setattr__("destroy_process_group", destroy_process_group)
torch.distributed.__setattr__("batch_isend_irecv", batch_isend_irecv)
torch.distributed.__setattr__("_shutdown_backend", _shutdown_backend)


def create_pg(dist_backend_opts, pg_opt):
    pg_options = ProcessGroupCNCL.Options()
    pg_options.group_name = dist_backend_opts.group_id
    pg_options._timeout = dist_backend_opts.timeout
    return torch_mlu._MLUC.ProcessGroupCNCL(
        dist_backend_opts.store,
        dist_backend_opts.group_rank,
        dist_backend_opts.group_size,
        pg_options,
    )


if not hasattr(torch.distributed.Backend, "CNCL"):
    torch.distributed.Backend.register_backend(
        "cncl", create_pg, extended_api=True, devices=["mlu"]
    )

if hasattr(torch_mlu._MLUC, "ProcessGroupCNCL"):
    from torch_mlu._MLUC import ProcessGroupCNCL

    torch.distributed.__setattr__("ProcessGroupCNCL", ProcessGroupCNCL)
    ProcessGroupCNCL.__module__ = "torch.distributed.distributed_c10d"
