import torch
import torch_mlu
import multiprocessing
from multiprocessing.reduction import ForkingPickler
from torch.multiprocessing.reductions import (shared_cache, rebuild_storage_filename,
                                              rebuild_storage_empty, rebuild_storage_fd,
                                              StorageWeakRef, fd_id, rebuild_cuda_tensor,
                                              rebuild_tensor, storage_from_cache,)
from torch._namedtensor_internals import check_serializing_named_tensor

def _reduce_storage(storage):

    if storage.is_cuda:
        raise RuntimeError(
            "Cannot pickle CUDA storage; try pickling a CUDA tensor instead"
        )
    if storage.is_mlu:
        raise RuntimeError(
            "Cannot pickle MLU storage; try pickling a MLU tensor instead"
        )
    elif torch.multiprocessing.get_sharing_strategy() == "file_system":
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        storage._shared_incref()
    elif storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd  # type: ignore[assignment]

    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage),) + metadata)

def rebuild_mlu_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(
            storage_cls, (storage_handle, storage_offset_bytes)
        )
        if storage is None:
            torch.mlu._lazy_init()
            storage = storage_cls._new_shared_mlu(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(
                storage
            )
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter_mlu(
                ref_counter_handle, ref_counter_offset, device=storage_device
            )

    _storage = (
        storage
        if (isinstance(storage, torch.UntypedStorage) or isinstance(storage, torch.mlu.UntypedStorage))
        else storage._untyped_storage
    )

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t

def _reduce_tensor(tensor):
    storage = tensor._typed_storage()

    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue)."
        )

    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)

    if storage._untyped_storage.device.type == "cuda":
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks purposely omitted here, see
        # Note [Don't serialize hooks]
        return (
            rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # tensor offset in its storage
                type(storage),
                tensor.dtype,
                device,
                handle,  # identifier which CUDA allocation is the storage in.
                storage_size_bytes,  # size(in bytes) of the storage
                storage_offset_bytes,  # offset(in bytes) of the storage in the CUDA allocation
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )

    if storage._untyped_storage.device.type == "mlu":
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_mlu_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        return (
            rebuild_mlu_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # tensor offset in its storage
                type(storage),
                tensor.dtype,
                device,
                handle,  # identifier which MLU allocation is the storage in.
                storage_size_bytes,  # size(in bytes) of the storage
                storage_offset_bytes,  # offset(in bytes) of the storage in the MLU allocation
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))

def rebuild_event_mlu(device, handle):
    return torch.mlu.Event.from_ipc_handle(device, handle)

def reduce_event_mlu(event):
    handle = event.ipc_handle()
    return (rebuild_event_mlu, (event.device, handle))

def apply_reductions_patch():
    ForkingPickler.register(torch.mlu.Event, reduce_event_mlu)
    torch.multiprocessing.reductions.reduce_storage = _reduce_storage
    torch.multiprocessing.reductions.reduce_tensor= _reduce_tensor

    torch.multiprocessing.reductions.init_reductions()