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

import functools
import queue
import threading
import warnings
from typing import Any
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, IterDataPipe, MapDataPipe
from torch.utils.data.dataloader import (
    _share_dist_seed,
    _get_distributed_settings,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter,
    _sharding_worker_init_fn,
    _DatasetKind,
)
from torch.utils.data import _utils


# for other backends, pin_memory_device need to set. if not set
# default behaviour is CUDA device. However, we want to change
# default device to CUDA/MLU to compatible with user's models.
def _BaseDataLoaderIter__init__(self, loader: DataLoader) -> None:
    self._dataset = loader.dataset
    self._shared_seed = None
    self._pg = None
    if isinstance(self._dataset, IterDataPipe):
        if dist.is_available() and dist.is_initialized():
            self._pg = dist.new_group(backend="gloo")
        self._shared_seed = _share_dist_seed(loader.generator, self._pg)
        shared_rng = torch.Generator()
        shared_rng.manual_seed(self._shared_seed)
        self._dataset = torch.utils.data.graph_settings.apply_random_seed(
            self._dataset, shared_rng
        )
    self._dataset_kind = loader._dataset_kind
    self._IterableDataset_len_called = loader._IterableDataset_len_called
    self._auto_collation = loader._auto_collation
    self._drop_last = loader.drop_last
    self._index_sampler = loader._index_sampler
    self._num_workers = loader.num_workers
    ws, rank = _get_distributed_settings()
    self._world_size = ws
    self._rank = rank
    # if pin_memory_device is selected and pin_memory is not set, the default behaviour false.
    if len(loader.pin_memory_device) == 0:
        self._pin_memory = loader.pin_memory and (
            torch.cuda.is_available() or torch.mlu.is_available()
        )
        self._pin_memory_device = None
    else:
        if not loader.pin_memory:
            warn_msg = (
                "pin memory device is set and pin_memory flag is not used then device pinned memory won't be used"
                "please set pin_memory to true, if you need to use the device pin memory"
            )
            warnings.warn(warn_msg)

        self._pin_memory = loader.pin_memory
        self._pin_memory_device = loader.pin_memory_device
    self._timeout = loader.timeout
    self._collate_fn = loader.collate_fn
    self._sampler_iter = iter(self._index_sampler)
    self._base_seed = (
        torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
    )
    self._persistent_workers = loader.persistent_workers
    self._num_yielded = 0
    self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"


# change default pin memory device to MLU/CUDA
def _MultiProcessingDataLoaderIter__init__(self, loader):
    _BaseDataLoaderIter.__init__(self, loader)

    self._prefetch_factor = loader.prefetch_factor

    assert self._num_workers > 0
    assert self._prefetch_factor > 0

    if loader.multiprocessing_context is None:
        multiprocessing_context = multiprocessing
    else:
        multiprocessing_context = loader.multiprocessing_context

    self._worker_init_fn = loader.worker_init_fn

    # Adds forward compatibilities so classic DataLoader can work with DataPipes:
    #   Additional worker init function will take care of sharding in MP and Distributed
    if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
        self._worker_init_fn = functools.partial(
            _sharding_worker_init_fn, self._worker_init_fn, self._world_size, self._rank
        )

    # No certainty which module multiprocessing_context is
    self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
    self._worker_pids_set = False
    self._shutdown = False
    self._workers_done_event = multiprocessing_context.Event()

    self._index_queues = []
    self._workers = []
    for i in range(self._num_workers):
        # No certainty which module multiprocessing_context is
        index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        # Need to `cancel_join_thread` here!
        # See sections (2) and (3b) above.
        index_queue.cancel_join_thread()
        w = multiprocessing_context.Process(
            target=_utils.worker._worker_loop,
            args=(
                self._dataset_kind,
                self._dataset,
                index_queue,
                self._worker_result_queue,
                self._workers_done_event,
                self._auto_collation,
                self._collate_fn,
                self._drop_last,
                self._base_seed,
                self._worker_init_fn,
                i,
                self._num_workers,
                self._persistent_workers,
                self._shared_seed,
            ),
        )
        w.daemon = True
        # NB: Process.start() actually take some time as it needs to
        #     start a process and pass the arguments over via a pipe.
        #     Therefore, we only add a worker to self._workers list after
        #     it started, so that we do not call .join() if program dies
        #     before it starts, and __del__ tries to join but will get:
        #     AssertionError: can only join a started process.
        w.start()
        self._index_queues.append(index_queue)
        self._workers.append(w)

    if self._pin_memory:
        self._pin_memory_thread_done_event = threading.Event()

        # Queue is not type-annotated
        self._data_queue = queue.Queue()  # type: ignore[var-annotated]
        if self._pin_memory_device == "xpu":
            current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
        elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
            custom_device_mod = getattr(torch, torch._C._get_privateuse1_backend_name())
            current_device = custom_device_mod.current_device()
        else:
            current_device = (
                torch.mlu.current_device()
                if torch.mlu.is_available()
                else torch.cuda.current_device()
            )  # choose cuda/mlu for default
        pin_memory_thread = threading.Thread(
            target=_utils.pin_memory._pin_memory_loop,
            args=(
                self._worker_result_queue,
                self._data_queue,
                current_device,
                self._pin_memory_thread_done_event,
                self._pin_memory_device,
            ),
        )
        pin_memory_thread.daemon = True
        pin_memory_thread.start()
        # Similar to workers (see comment above), we only register
        # pin_memory_thread once it is started.
        self._pin_memory_thread = pin_memory_thread
    else:
        self._data_queue = self._worker_result_queue  # type: ignore[assignment]

    # In some rare cases, persistent workers (daemonic processes)
    # would be terminated before `__del__` of iterator is invoked
    # when main process exits
    # It would cause failure when pin_memory_thread tries to read
    # corrupted data from worker_result_queue
    # atexit is used to shutdown thread and child processes in the
    # right sequence before main process exits
    if self._persistent_workers and self._pin_memory:
        import atexit

        for w in self._workers:
            atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

    # .pid can be None only before process is spawned (not the case, so ignore)
    _utils.signal_handling._set_worker_pids(id(self), tuple(w.pid for w in self._workers))  # type: ignore[misc]
    _utils.signal_handling._set_SIGCHLD_handler()
    self._worker_pids_set = True
    self._reset(loader, first_iter=True)


# only for cnnl gencase, dump data when TORCH_MLU_COUNTER_START and
# TORCH_MLU_COUNTER_END are set
def _BaseDataLoaderIter__next__(self) -> Any:
    if torch.mlu.is_available():
        from torch_mlu.utils.counter import (
            _GENCASE_ENABLED,
            _update_and_check_for_gencase,
        )

        if _GENCASE_ENABLED:
            _update_and_check_for_gencase()

    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        data = self._next_data()
        self._num_yielded += 1
        if (
            self._dataset_kind == _DatasetKind.Iterable
            and self._IterableDataset_len_called is not None
            and self._num_yielded > self._IterableDataset_len_called
        ):
            warn_msg = (
                "Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                "samples have been fetched. "
            ).format(self._dataset, self._IterableDataset_len_called, self._num_yielded)
            if self._num_workers > 0:
                warn_msg += (
                    "For multiprocessing data-loading, this could be caused by not properly configuring the "
                    "IterableDataset replica at each worker. Please see "
                    "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."
                )
            warnings.warn(warn_msg)
        return data


def apply_dataloader_patch():
    _BaseDataLoaderIter.__init__ = _BaseDataLoaderIter__init__
    _BaseDataLoaderIter.__next__ = _BaseDataLoaderIter__next__
    _MultiProcessingDataLoaderIter.__init__ = _MultiProcessingDataLoaderIter__init__
