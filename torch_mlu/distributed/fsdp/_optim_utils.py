import functools
import logging
from typing import (
    Any,
    cast,
    Dict,
    List,
)

import torch
import torch.distributed as dist
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._optim_utils import (
    FSDPParamInfo,
    StateInfo,
    _convert_all_state_info,
    _unflatten_orig_param_states,
)

logger = logging.getLogger(__name__)


def _allgather_orig_param_states(
    fsdp_param_info: FSDPParamInfo,
    gathered_state_info: List[Dict[str, StateInfo]],
    input_states: Dict[str, Any],
    shard_state: bool,
    to_save: bool,
    cpu_offload: bool,
) -> Dict[str, Dict[str, Any]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API allgathers
    all tensor states and restore non-tensor states from ``gathered_state_info``.
    """
    fsdp_state = fsdp_param_info.state
    if fsdp_state.rank == 0 and dist.get_debug_level() == dist.DebugLevel.DETAIL:
        logger.warning(
            "MLU Memory Summary before calling to _allgather_orig_param_states %s",
            torch.mlu.memory_summary(),
        )

    output_states: Dict[str, Dict[str, Any]] = {fqn: {} for fqn in input_states.keys()}

    dtype, state_buffers = _convert_all_state_info(
        fsdp_param_info, gathered_state_info, input_states, output_states
    )

    if len(state_buffers) == 0:
        return output_states

    has_state_params: List[bool] = [
        True if fqn in output_states else False
        for fqn, idx in fsdp_param_info.param_indices.items()
    ]

    # Loop through the ``state_buffers`` and construct the flattened, concatenated,
    # sharded states. The size of the constructed state will be the same size as
    # flat_param (also sharded).
    # Then we perform an allgather_into_tensor to get the full flat_param state.
    # The full flat_param state is the result of concatenation of multiple states
    # the order of of flat_param._fqns.
    # The final step is to split the flat_param state into original param states
    # and return the result.
    flat_param = fsdp_param_info.handle.flat_param
    empty_func = functools.partial(
        torch.empty, dtype=dtype, device=fsdp_state.compute_device
    )
    gathered_tensor = empty_func(flat_param._padded_unsharded_size)
    # Synchronize can be slow but this will be easier for us to debug.
    torch.mlu.synchronize()
    for state_name, buffers in state_buffers.items():
        local_buffers: List[torch.Tensor] = []
        begin = fsdp_state.rank * flat_param._sharded_size.numel()
        # End is inclusive.
        end = begin + flat_param._sharded_size.numel() - 1
        # param_idx corresponds to the parameter index in the FlatParameter.
        mem_offset, param_idx = 0, 0
        for numel, is_padding in zip(
            flat_param._numels_with_padding, flat_param._is_padding_mask
        ):
            frozen_and_no_state = not is_padding and (
                not fsdp_param_info.param_requires_grad[param_idx]
                and not has_state_params[param_idx]
            )

            if is_padding or frozen_and_no_state:
                # This memory range is a padding or the param is frozen and does
                # not require gradient. For the later case, we treat it as a
                # padding and add empty values to the local_buffers.

                padding_begin, padding_end = mem_offset, mem_offset + numel - 1
                if padding_begin <= begin <= padding_end:
                    # The range is an align padding before the first parameter in
                    # the shard. The shard includes parts of this align padding.
                    padding_len = (
                        padding_end - begin + 1
                        if end >= padding_end
                        else end - begin + 1
                    )
                elif padding_begin <= end <= padding_end:
                    # The range is an align padding after the last parameter in
                    # the shard. The shard includes parts of this align padding.
                    padding_len = (
                        end - padding_begin + 1
                        if begin <= padding_begin
                        else end - begin + 1
                    )
                elif begin < padding_begin <= padding_end < end:
                    # The range is an align padding that is completely in the
                    # shard.
                    padding_len = numel
                else:
                    padding_len = 0
                if padding_len:
                    local_buffers.append(empty_func(padding_len))

            if not is_padding:
                # This memory range is a parameter in FlatParameter. So there
                # should be an corresponding state in the optimizer unless the
                # parameter is frozen, which we treat it as a padding above.

                # We need to check if this rank owns the buffer. If this is None:
                # 1.) the rank does not own any part of the original parameter.
                #     As a result, there is no corresponding optimizer state on
                #     the rank as well.
                # 2.) the parameter is frozen AND no optimizer state for the
                #     parameter. If a parameter is frozen, there can still be
                #     optimizer state if the parameter is not frozen in the
                #     previous steps.
                if buffers[param_idx] is not None:
                    local_buffers.append(cast(torch.Tensor, buffers[param_idx]))
                param_idx += 1

            mem_offset += numel

        shard_numel_padded = flat_param._sharded_size.numel() - (
            sum(t.numel() for t in local_buffers)
        )

        assert flat_param._shard_numel_padded == shard_numel_padded, (
            "Manually calculated _sharded_numel_padded is incorrect. "
            f"_shard_numel_padded={flat_param._shard_numel_padded}, "
            f"shard_numel_padded={shard_numel_padded}, "
            f"_sharded_size.numel={flat_param._sharded_size.numel()}, "
            f"_numels_with_padding={flat_param._numels_with_padding}, "
            f"begin={begin}, end={end},"
        )
        if shard_numel_padded > 0:
            # Add right-handed padding.
            local_buffers.append(empty_func(shard_numel_padded))
        local_shard = torch.cat(local_buffers)
        assert local_shard.numel() * fsdp_state.world_size == gathered_tensor.numel(), (
            "The size of local shard times the world size should equal to the "
            "gathered tensor size. The inconsistency may be from a bug of "
            "FlatParameter's metadata or the reconstruction logic in optimizer "
            "state dict."
        )
        torch.mlu.synchronize()
        with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER):
            dist.all_gather_into_tensor(
                gathered_tensor, local_shard, group=fsdp_state.process_group
            )
            # Synchronize can be slow but this will be easier for us to debug.
            torch.mlu.synchronize()

        unpadded_tensor = gathered_tensor[: flat_param._unpadded_unsharded_size.numel()]
        flat_param_handle = fsdp_param_info.handle
        orig_states = flat_param_handle._get_unflat_views_aligned(unpadded_tensor)
        assert len(orig_states) == len(fsdp_param_info.param_indices), (
            "The number of parameters from FlatParameter is not consistent to "
            "the number of states used by optimizer state dict reconstruction "
            "logic."
        )
        for fqn, idx in fsdp_param_info.param_indices.items():
            if fsdp_param_info.param_requires_grad[idx] or fqn in output_states:
                output_states[fqn][state_name] = orig_states[idx]

        _unflatten_orig_param_states(
            fsdp_param_info,
            output_states,
            state_name,
            shard_state,
            to_save,
            cpu_offload,
        )

    del gathered_tensor
    return output_states


def apply_optim_utils_patch():
    torch.distributed.fsdp._optim_utils._allgather_orig_param_states = (
        _allgather_orig_param_states
    )
