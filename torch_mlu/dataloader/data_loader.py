import os
import warnings
import torch
import torch.utils.data.dataloader as dataloader
import torch.utils.data._utils as _utils
import torch.utils.data._utils.worker as worker
from torch._six import string_classes
import collections.abc as container_abcs
from torch.utils.data.dataloader import _DatasetKind
import torch_mlu


class _MLUMultiProcessingDataLoaderIter(dataloader._MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        super(_MLUMultiProcessingDataLoaderIter, self).__init__(loader)

        self._use_io_queue = bool(
            (os.getenv("USE_IO_QUEUE") is not None)
            and (os.getenv("USE_IO_QUEUE").upper() in ["ON", "1", "YES", "TRUE", "Y"])
        )
        if self._use_io_queue:
            warnings.warn("using io_queue in data_loader will take more memory on MLU.")
            self._mlu_data = None
            self._notifier = torch.mlu.Event()
            self._io_queue = torch.mlu.Stream()
            self._current_queue = None
            self._mlu_rcvd_idx = (
                0  # idx of the next task to be returned in __next__ when using io queue
            )

        if self._use_io_queue:
            self._mlu_data = self._next_mlu_data()

    def _next_mlu_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if (
                    len(info) == 2 or self._workers_status[worker_id]
                ):  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            data_to_mlu = None
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._data_to_mlu(self._process_data(data))

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1

            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._data_to_mlu(self._process_data(data))

    def _data_to_mlu(self, data):
        with torch.mlu.stream(self._io_queue):
            data = self._to_mlu(data)
            self._notifier.record()
        return data

    def _to_mlu(self, data):
        if isinstance(data, torch.Tensor):
            return data.to("mlu", non_blocking=True)
        elif isinstance(data, string_classes):
            return data
        elif isinstance(data, container_abcs.Mapping):
            return {k: self._to_mlu(sample) for k, sample in data.items()}
        elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
            return type(data)(*(self._to_mlu(sample) for sample in data))
        elif isinstance(data, container_abcs.Sequence):
            return [self._to_mlu(sample) for sample in data]
        elif hasattr(data, "to"):
            return data.to("mlu", non_blocking=True)
        else:
            return data

    def _next_data(self):
        if self._use_io_queue:
            self._notifier.wait()
            return_data = self._mlu_data
            if self._rcvd_idx < self._send_idx:
                self._mlu_data = self._next_mlu_data()
            else:
                self._shutdown_workers()
            if self._mlu_rcvd_idx >= self._rcvd_idx:
                raise StopIteration
            self._mlu_rcvd_idx += 1
            return return_data
        else:
            return super()._next_data()
