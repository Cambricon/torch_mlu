import io
import torch
from torch_mlu import _MLUC
from torch._utils import _type, _get_async_or_non_blocking
from typing import Any, TypeVar, Type, cast, Union, Optional as _Optional
from torch.types import Storage
from torch.storage import _StorageBase, _load_from_bytes, _share_memory_lock_protected


def _mlu(self, device=None, non_blocking=False, **kwargs):
    """Returns a copy of this object in MLU memory.

    If this object is already in MLU memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination MLU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    """
    non_blocking = _get_async_or_non_blocking("mlu", non_blocking, kwargs)
    if self.is_mlu:
        if device is None:
            device = torch.mlu.current_device()
        if self.get_device() == device:
            if self.__class__.__name__ == "UntypedStorage":
                return self
            else:
                return self.untyped()
    else:
        if device is None:
            device = -1
    with torch.mlu.device(device):
        if self.is_sparse:
            raise RuntimeError("mlu does not support sparse tensor")
        else:
            if self.__class__.__name__ == "UntypedStorage":
                untyped_storage = torch.mlu.UntypedStorage(
                    self.size(), device=torch.device("mlu")
                )
                untyped_storage.copy_(self, non_blocking)
                return untyped_storage
            else:
                untyped_storage = torch.mlu.UntypedStorage(
                    self.nbytes(), device=torch.device("mlu")
                )
                untyped_storage.copy_(self.untyped(), non_blocking)
                return untyped_storage


T = TypeVar("T", bound="Union[_MLUStorageBase]")


class _MLUStorageBase(object):
    _cdata: Any
    is_sparse: bool = False
    is_sparse_csr: bool = False
    device: torch.device

    def __init__(self, *args, **kwargs):
        ...  # noqa: E704

    def __len__(self) -> int:
        ...  # noqa: E704

    def __getitem__(self, idx):
        ...  # noqa: E704

    def __setitem__(self, *args, **kwargs):
        ...  # noqa: E704

    def copy_(self, source: T, non_blocking: _Optional[bool] = None) -> T:
        ...  # noqa: E704

    def new(self) -> T:
        ...  # type: ignore[empty-body, misc, type-var] # noqa: E704

    def nbytes(self) -> int:
        ...  # noqa: E704

    def size(self) -> int:
        return self.nbytes()

    def type(self, dtype: _Optional[str] = None, non_blocking: bool = False) -> T:
        ...  # noqa: E704

    def mlu(self, device=None, non_blocking=False, **kwargs) -> T:
        ...  # noqa: E704

    def element_size(self) -> int:
        ...  # noqa: E704

    def get_device(self) -> int:
        return self.device.index

    def data_ptr(self) -> int:
        ...  # noqa: E704

    def _share_filename_cpu_(self, *args, **kwargs):
        ...  # noqa: E704

    def _share_fd_cpu_(self, *args, **kwargs):
        ...  # noqa: E704

    @classmethod
    def _new_using_filename_cpu(cls: Type[T], size: int) -> T:
        ...  # noqa: E704

    @classmethod
    def _new_using_fd_cpu(cls: Type[T], size: int) -> T:
        ...  # noqa: E704

    @classmethod
    def from_buffer(cls, *args, **kwargs) -> T:
        ...  # noqa: E704

    @classmethod
    def _new_shared_filename_cpu(
        cls, manager, obj, size, *, device=None, dtype=None
    ) -> T:
        ...  # noqa: E704

    @classmethod
    def _release_ipc_counter_mlu(cls: Type[T], *args, **kwargs) -> T: ...  # type: ignore[empty-body] # noqa: E704
    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs) -> T:
        ...  # noqa: E704

    def _shared_decref(self) -> T:
        ...  # noqa: E704

    def _write_file(self, *args, **kwargs):
        ...  # noqa: E704

    def resize_(self, size: int):
        ...  # noqa: E704

    def _weak_ref(self, *args, **kwargs) -> T:
        ...  # noqa: E704

    def _set_from_file(self, *args, **kwargs):
        ...  # noqa: E704

    def _set_cdata(self, *args, **kwargs):
        ...  # noqa: E704

    def _share_mlu_(self, *args, **kwargs):
        ...  # noqa: E704

    def is_shared(self) -> bool:
        ...  # noqa: E704

    @classmethod
    def _new_shared_mlu(cls: Type[T], *args, **kwargs) -> T: ...  # type: ignore[empty-body] # noqa: E704
    def _shared_incref(self, *args, **kwargs): ...  # noqa: E704
    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        ...  # noqa: E704

    @property
    def is_mlu(self):
        ...  # noqa: E704

    @classmethod
    def from_file(cls, filename, shared, nbytes) -> T:
        ...  # noqa: E704

    @classmethod
    def _expired(cls, *args, **kwargs) -> T:
        ...  # noqa: E704

    def _byteswap(self, *args, **kwargs):
        ...  # noqa: E704

    def __str__(self):
        info_str = (
            f"[{torch.typename(self)}(device={self.device}) " f"of size {len(self)}]"
        )
        if self.device.type == "meta":
            return "...\n" + info_str
        else:
            data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
            return data_str + "\n" + info_str

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], range(self.size())))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault("torch", {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def __sizeof__(self):
        return super().__sizeof__() + self.size()

    def clone(self):
        """Returns a copy of this storage"""
        return type(self)(self.nbytes(), device=self.device).copy_(self)

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        return list(self)

    def cpu(self):
        """Returns a CPU copy of this storage if it's not already on the CPU"""
        if self.device.type != "cpu":
            return torch.UntypedStorage(self.size()).copy_(self, False)
        else:
            return self

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .to(dtype)
            .storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type"""
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type"""
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type"""
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type"""
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type"""
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type"""
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type"""
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type"""
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type"""
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type"""
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type"""
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type"""
        return self._to(torch.cfloat)

    def is_pinned(self, device: Union[str, torch.device] = "mlu"):
        r"""Determine whether the CPU storage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'mlu'``.

        Returns:
            A boolean variable.
        """
        return (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .is_pinned(device)
        )

    def pin_memory(self, device: Union[str, torch.device] = "mlu"):
        r"""Copies the CPU storage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'mlu'``.

        Returns:
            A pinned CPU storage.
        """
        if self.device.type != "cpu":
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")

        pinned_tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .pin_memory(device)
        )
        return pinned_tensor.untyped_storage()

    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for MLU
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        """
        from torch.multiprocessing import get_sharing_strategy

        if self.is_mlu:
            pass  # MLU doesn't use POSIX shared memory
        elif get_sharing_strategy() == "file_system":
            self._share_filename_cpu_()
        else:
            self._share_fd_cpu_()
        return self

    @classmethod
    def _new_shared(cls, size, *, device="cpu"):
        """Creates a new storage in shared memory with the same data type"""
        from torch.multiprocessing import get_sharing_strategy

        device = torch.device(device)
        if device.type == "mlu":
            return cls(size, device=device)
        elif get_sharing_strategy() == "file_system":
            return cls._new_using_filename_cpu(size)
        else:
            return cls._new_using_fd_cpu(size)

    def untyped(self):
        return self

    def byteswap(self, dtype):
        """Swaps bytes in underlying data"""
        elem_size = torch._utils._element_size(dtype)
        # for complex types, don't swap first and second numbers
        if dtype.is_complex:
            elem_size = max(int(elem_size / 2), 1)
        self._byteswap(elem_size)


_MLUStorageBase.type = _type  # type: ignore[assignment]
_MLUStorageBase.mlu = _mlu  # type: ignore[assignment]
_StorageBase.mlu = _mlu  # type: ignore[assignment]


class UntypedStorage(_MLUC.StorageBase, _MLUStorageBase):
    def __getitem__(self, *args, **kwargs):
        if self.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")
        return super().__getitem__(*args, **kwargs)

    @property
    def is_mlu(self):
        return self.device.type == "mlu"

    @_share_memory_lock_protected
    def share_memory_(self, *args, **kwargs):
        return super().share_memory_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs):
        return super()._share_fd_cpu_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs):
        return super()._share_filename_cpu_(*args, **kwargs)
