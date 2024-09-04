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
from torch._utils import classproperty
import torch_mlu
from torch_mlu.mlu._utils import _LazySeedTracker
from .reductions import apply_reductions_patch

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

def is_available():
    r"""Returns a bool indicating if MLU is currently available."""
    return torch.mlu.device_count() > 0

def is_bf16_supported():
    r"""Returns a bool indicating if MLU is currently support bf16."""
    return torch.mlu.get_device_properties(torch.mlu.current_device()).major >= 5

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
