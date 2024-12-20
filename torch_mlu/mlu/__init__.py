# All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
# All rights reserved.
# All other contributions:
# Copyright (c) 2014--2022, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import threading
import traceback
import os
from typing import cast
import binascii
from collections import namedtuple
import functools

import torch
from torch._utils import classproperty
import torch_mlu
from torch_mlu.mlu._utils import _LazySeedTracker
from .reductions import apply_reductions_patch
from typing import Union, Tuple, Optional, TypeVar, Any, List
from torch import device as _device

from .graphs import (
    MLUGraph,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)

from .streams import Event, ExternalStream, Stream
from .cncl import *
from .autocast_utils import *
from . import amp
from ._utils import _get_device_index

try:
    from torch_mlu._MLUC import _cnrt
except ImportError:
    _cnrt = None

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch_mlu._MLUC, "_mlu_isInBadFork", lambda: False)

OutOfMemoryError = torch._C._OutOfMemoryError

_lazy_seed_tracker = _LazySeedTracker()

def _is_compiled() -> bool:
    r"""Return true if compile with MLU support."""
    return True

### lazy_init
def init():
    r"""Initialize PyTorch's MLU state.
    """
    _lazy_init()

def is_initialized():
    r"""Returns whether PyTorch's MLU state has been initialized."""
    return _initialized and not _is_in_bad_fork()

def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # Don't store the actual traceback to avoid memory cycle
            _queued_calls.append((callable, traceback.format_stack()))

def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                    "Cannot re-initialize MLU in forked subprocess. To use MLU with multiprocessing, you must use the "
                    "'spawn' start method")
        torch_mlu._MLUC._mlu_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = ("MLU call failed lazily at initialization with error: {}\n\n"
                    "MLU call was originally invoked at:\n\n{}").format(str(e), orig_traceback)
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True


default_generators: Tuple[torch._C.Generator] = ()


class device:
    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch.mlu.current_device()
        if self.idx != self.prev_idx:
            torch.mlu.set_device(self.idx)
        if not torch.jit.is_scripting():
            torch.mlu._lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch.mlu.set_device(self.prev_idx)
        return False


Device = device
_device_t = Union[_device, str, int, None]


### Device management
def set_device(device):
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``MLU_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device, optional=True)
    if device >= 0:
        torch_mlu._MLUC._mlu_setDevice(device)


def current_device():
    r"""Returns the index of a currently selected device."""
    torch.mlu._lazy_init()
    return torch_mlu._MLUC._mlu_getDevice()


def get_device_properties(device: _device_t):
    r"""Gets the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _MLUDeviceProperties: the properties of the device
    """
    torch.mlu._lazy_init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= torch.mlu.device_count():
        raise AssertionError("Invalid device id")
    return torch_mlu._MLUC._get_device_properties(device)


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Gets the cuda capability of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.mlu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device. eg. MLU370
    """
    return get_device_properties(device).name


@functools.lru_cache(32)
def supports_linear_memory(device: Optional[_device_t] = None) -> bool:
    r"""Returns a boolean indicating if the device supports linear memory.
    Args:
        device (torch.device or int, optional): device for which to return the
            property. It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return get_device_properties(device).supports_linear_memory


def synchronize(device: Optional[_device_t] = None):
    r"""Waits for all kernels in all streams on a MLU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    torch.mlu._lazy_init()
    device_index = _get_device_index(device, optional=True)
    with Device(device_index):
        torch_mlu._MLUC._synchronize()


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use tensors as arguments. If a given object is
    not allocated on a MLU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.device.type == "mlu" else -1
        super(device_of, self).__init__(idx)


def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    r"""Checks if peer access between two devices is possible."""
    torch.mlu._lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= torch.mlu.device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= torch.mlu.device_count():
        raise AssertionError("Invalid peer device id")
    return torch_mlu._MLUC._mlu_canDeviceAccessPeer(device, peer_device)


def cnrt():
    _lazy_init()
    return _cnrt

class CnrtError(RuntimeError):
    def __init__(self, code: int) -> None:
        msg = _cnrt.mluGetErrorStr(_cnrt.mluError(code))
        super().__init__(f"{msg} ({code})")

def check_error(res: int) -> None:
    if res != _cnrt.mluError.success:
        raise CnrtError(res)

def _parse_visible_devices():
    r"""Parse CN_VISIBLE_DEVICES/MLU_VISIBLE_DEVICES environment variable. Keep align with cnrt"""
    var = os.getenv("CN_VISIBLE_DEVICES")

    if var is None:
        var = os.getenv("MLU_VISIBLE_DEVICES")

    if var is None:
        var = os.getenv("CUDA_VISIBLE_DEVICES")

    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with."""
        if not s:
            return -1
        if s.isdigit():
            return int(s)
        else:
            return -1

    def parse_list_from_uuid(lst):
        rcs = []
        for elem in lst.split(","):
            # Repeated id results in empty set
            if elem in rcs:
                return []
            if "-" not in elem:
                break
            rcs.append(elem)
        return rcs

    if "-" in var and len(var) >= 36:
        return parse_list_from_uuid(var)
    rc = []
    for elem in var.split(","):
        x = _strtoul(elem.strip().lstrip())
        # Repeated ordinal results in empty set
        if x in rc:
            return []
        # Negative value aborts the sequence
        if x < 0:
            break
        rc.append(x)
    return rc

def _raw_device_count_cndev():
    r"""Return number of devices as reported by CNDEV or negative value if CNDEV discovery/initialization failed."""
    from ctypes import byref, c_int, c_uint, Structure, CDLL

    class CardInfo(Structure):
        _fields_ = [("version", c_int),
                    ("number", c_uint)]

    cndev_h = CDLL("libcndev.so")
    reserved = c_int(0)
    rc = cndev_h.cndevInit(reserved)
    if rc != 0:
        warnings.warn("Can't initialize CNDEV")
        return -1
    card_info = CardInfo(6, 0)
    rc = cndev_h.cndevGetDeviceCount(byref(card_info))
    if rc != 0:
        warnings.warn("Can't get cndev device count")
        return -1
    del cndev_h
    return card_info.number

def _transform_uuid_to_ordinals(candidates, uuids):
    r"""Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials IDs."""

    def uuid_to_orinal(candidate, uuids) -> int:
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if uuid == candidate:
                best_match = idx
        return best_match

    rc = []
    for candidate in candidates:
        idx = uuid_to_orinal(candidate, uuids)
        # First invalid ordinal stops parsing
        if idx < 0:
            break
        # Duplicates result in empty set
        if idx in rc:
            return []
        rc.append(idx)
    return rc


def _raw_device_uuid_cndev():
    r"""Return list of device UUID as reported by CNDEV or None if CNDEV discovery/initialization failed."""
    from ctypes import byref, c_int, c_uint, c_uint8, c_uint64, CDLL, Structure

    class CardInfo(Structure):
        _fields_ = [("version", c_int),
                    ("number", c_uint)]

    class UUID(Structure):
        _fields_ = [("version", c_int),
                      ("uuid", c_uint8 * 37),
                      ("ncsUUID64", c_uint64)]

    cndev_h = CDLL("libcndev.so")
    reserved = c_int(0)
    rc = cndev_h.cndevInit(reserved)
    if rc != 0:
        warnings.warn("Can't initialize CNDEV")
        return None
    card_info = CardInfo(6, 0)
    rc = cndev_h.cndevGetDeviceCount(byref(card_info))
    if rc != 0:
        warnings.warn("Can't get cndev device count")
        return None
    uuids = []
    for idx in range(card_info.number):
        device_id = c_int()
        rc = cndev_h.cndevGetDeviceHandleByIndex(idx, byref(device_id))
        if rc != 0:
            warnings.warn("Can't get device handle")
            return None
        uuid_len = 37
        buf = c_uint8 * uuid_len
        uuid = UUID(6, buf(), 0)
        rc = cndev_h.cndevGetUUID(byref(uuid), device_id)
        if rc != 0:
            warnings.warn("Can't get device UUID")
            return None
        # uuids.append(str(bytearray(uuid.uuid)))
        uuids.append(''.join([chr(i) for i in uuid.uuid]).rstrip('\x00'))
    del cndev_h
    return uuids

def _device_count_cndev():
    r"""Return number of devices as reported by CNDEV taking MLU_VISIBLE_DEVICES into account.

    Negative value is returned if CNDEV discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            uuids = _raw_device_uuid_cndev()
            if uuids is None:
                return -1
            visible_devices = _transform_uuid_to_ordinals(
                cast([], visible_devices), uuids
            )
        else:
            raw_cnt = _raw_device_count_cndev()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            for idx, val in enumerate(visible_devices):
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)



_cached_device_count = None

def device_count():
    r"""Return the number of MLUs available."""
    global _cached_device_count
    if _cached_device_count is not None:
        return _cached_device_count
    cndev_count = _device_count_cndev()
    r = torch_mlu._MLUC._mlu_getDeviceCount() if cndev_count < 0 else cndev_count
    # NB: Do not cache the device count prior to MLU initialization, because
    # the number of devices can change due to changes to MLU_VISIBLE_DEVICES
    # setting prior to MLU initialization.
    if _initialized:
        _cached_device_count = r
    return r

def _cndev_based_avail():
    var = os.getenv("PYTORCH_CNDEV_BASED_MLU_CHECK")

    if var is None:
        var = os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK")

    return var == "1"

def is_available():
    r"""Returns a bool indicating if MLU is currently available."""
    if _cndev_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using CNDEV at the cost of a weaker MLU availability assessment. Note that if CNDEV initialization
        # fails, this assessment falls back to the default CN Runtime API assessment (`cnrtDeviceCount`)
        return device_count() > 0
    else:
        # This uses the CN Runtime API `cnrtGetDeviceCount` which in turn initializes the MLU Driver
        # API via `cnInit`
        return torch_mlu._MLUC._mlu_getDeviceCount() > 0

def is_bf16_supported():
    r"""Returns a bool indicating if MLU is currently support bf16."""
    return torch.mlu.get_device_properties(torch.mlu.current_device()).major >= 5

def _sleep(cycles):
    torch_mlu._MLUC._mlu_sleep(cycles)

from .memory import *
from .random import *


################################################################################
# Define Storage
################################################################################

from ..storage import UntypedStorage
from torch.storage import _LegacyStorage, _warn_typed_storage_removal

class _MluLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError('from_buffer: Not available for MLU storage')

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError('_new_with_weak_ptr: Not available for MLU storage')

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError('_new_shared_filename: Not available for MLU storage')

class ByteStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_MluLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat

del _LegacyStorage
del _MluLegacyStorage

_mlu_storage_classes = [
    UntypedStorage, DoubleStorage, FloatStorage, LongStorage,
    IntStorage, ShortStorage, CharStorage, ByteStorage,
    HalfStorage, BoolStorage, BFloat16Storage, ComplexDoubleStorage, ComplexFloatStorage]
for r in _mlu_storage_classes:
    torch._storage_classes.add(r)

def ipc_collect():
    _lazy_init()
    return torch_mlu._MLUC._mlu_ipc_collect()


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.mlu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch.mlu._lazy_init()
    streamdata = torch_mlu._MLUC._mlu_getCurrentMLUStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.mlu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch.mlu._lazy_init()
    streamdata = torch_mlu._MLUC._mlu_getDefaultStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


StreamInfo = namedtuple("StreamInfo", ["cncl_stream", "clique_id"])


def cncl_stream(device: Optional[_device_t] = None) -> List[StreamInfo]:
    r"""Returns a list of MLU :class:`Stream` used by CNCL for a given device.

    To distinguish between different streams in the list, each stream is
    bundled with a unique CliqueId into a :namedtuple:`StreamInfo`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            a list of  MLU :class:`Stream` used by CNCL for the current device,
            given by :func:`~torch.mlu.current_device`, if :atta:`device` is
            ``None``(default)

    Returns:
        List[StreamInfo]: A list of :namedtuple:`StreamInfo`, each containing:
            - cncl_stream (Stream): the MLU :class:`Stream` used by CNCL.
            - clique_id (str): Unique CliqueId bundled with cncl_stream.

    """
    torch.mlu._lazy_init()
    streams_data = torch_mlu._MLUC._mlu_getCnclStream(
        _get_device_index(device, optional=True)
    )
    stream_info_list = []
    for stream_data in streams_data:
        cncl_stream = Stream(
            stream_id=stream_data[0],
            device_index=stream_data[1],
            device_type=stream_data[2],
        )
        clique_id_raw = stream_data[3]
        clique_id_data = binascii.hexlify(clique_id_raw).decode("utf-8")
        stream_info_list.append(
            StreamInfo(cncl_stream=cncl_stream, clique_id=clique_id_data[:16])
        )

    return stream_info_list


def set_stream(stream: Stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch_mlu._MLUC._mlu_setStream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


class StreamContext(object):
    r"""Context-manager that selects a given stream.

    All MLU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch.mlu.Stream"]

    def __init__(self, stream: Optional["torch.mlu.Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mlu.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mlu.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or MLU device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.mlu.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with torch.mlu.device(cur_stream.device):
                self.dst_prev_stream = torch.mlu.current_stream(cur_stream.device)
        torch.mlu.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no MLU device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.mlu.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.mlu.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.mlu.Stream"]) -> StreamContext:
    return StreamContext(stream)

from . import amp, cnpx, profiler

__all__ = [
    # Typed storage and tensors
    "BFloat16Storage",
    "BFloat16Tensor",
    "BoolStorage",
    "BoolTensor",
    "ByteStorage",
    "ByteTensor",
    "CharStorage",
    "CharTensor",
    "ComplexDoubleStorage",
    "ComplexFloatStorage",
    "DoubleStorage",
    "DoubleTensor",
    "FloatStorage",
    "FloatTensor",
    "HalfStorage",
    "HalfTensor",
    "IntStorage",
    "IntTensor",
    "LongStorage",
    "LongTensor",
    "ShortStorage",
    "ShortTensor",
    "MLUGraph",
    "Event",
    "ExternalStream",
    "OutOfMemoryError",
    "Stream",
    "StreamContext",
    "amp",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "can_device_access_peer",
    "check_error",
    "cnrt",
    "current_device",
    "current_stream",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "device_of",
    "empty_cache",
    "MLUPluggableAllocator",
    "change_current_allocator",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_rng_state",
    "get_rng_state_all",
    "graph",
    "graph_pool_handle",
    "graphs",
    "init",
    "initial_seed",
    "ipc_collect",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "cncl",
    "cnpx",
    "random",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "stream",
    "streams",
    "synchronize",
    "profiler",
]
