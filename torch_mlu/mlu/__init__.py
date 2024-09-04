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
import torch
from torch._utils import classproperty, _LazySeedTracker
import torch_mlu

from .device import *
from .graphs import (
    MLUGraph,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)

from .streams import *
from .cncl import *
from .autocast_utils import *
from . import amp
from .storage import *
from .reductions import apply_reductions_patch

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch_mlu._MLUC, "_mlu_isInBadFork", lambda: False)

OutOfMemoryError = torch._C.OutOfMemoryError

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

def _parse_visible_devices():
    r"""Parse CN_VISIBLE_DEVICES/MLU_VISIBLE_DEVICES environment variable. Keep align with cnrt"""
    var = os.getenv("CN_VISIBLE_DEVICES")

    if var is None:
        var = os.getenv("MLU_VISIBLE_DEVICES")

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
    mlu_env = os.getenv("PYTORCH_CNDEV_BASED_MLU_CHECK", None)
    if mlu_env:
        return mlu_env == "1"
    return os.getenv("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"

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

from .memory import *
from .random import *

################################################################################
# Define Storage
################################################################################

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

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_mlu(*args, **kwargs)

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
    DoubleStorage, FloatStorage, LongStorage,
    IntStorage, ShortStorage, CharStorage, ByteStorage,
    HalfStorage, BoolStorage, BFloat16Storage, ComplexDoubleStorage, ComplexFloatStorage]
for r in _mlu_storage_classes:
    torch._storage_classes.add(r)

def ipc_collect():
    r"""Force collects MLU memory after it has been released by MLU IPC.

    .. note::
        Checks if any sent MLU tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    """
    _lazy_init()
    return torch_mlu._MLUC._mlu_ipc_collect()
