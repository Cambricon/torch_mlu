import ctypes
import torch
import torch_mlu
from typing import Union, Tuple, Optional, TypeVar, Any, List
from collections import namedtuple
import binascii
from torch.cuda._utils import _dummy_type
from .device import _device_t
from ._utils import _get_device_index

if not hasattr(torch_mlu._MLUC, "_MLUStreamBase"):
    # Define dummy base classes
    torch_mlu._MLUC.__dict__["_MLUStreamBase"] = _dummy_type("_MLUStreamBase")


# define Stream
class Stream(torch_mlu._MLUC._MLUStreamBase):
    r"""Wrapper around a MLU stream.

    A MLU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, should be 0 or
            negative, where negative numbers indicate higher priority. By default,
            streams have priority 0.

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)
        else:
            with torch.mlu.device(device):
                return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        r"""Makes all future work submitted to the stream wait for an event.

        Args:
            event (torch.mlu.Event): an event to wait for.

        .. note:: This is a wrapper around ``cnrtQueueWaitNotifier()``: see
           `MLU Queue documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        """
        event.wait(self)

    def wait_stream(self, stream):
        r"""Synchronizes with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        r"""Records an event.

        Args:
            event (torch.mlu.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = torch.mlu.Event()
        event.record(self)
        return event

    def query(self):
        r"""Checks if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed."""
        return super(Stream, self).query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cnrtQueueSync()``
        """
        super(Stream, self).synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.mlu_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super(Stream, self).__eq__(o)
        return False

    def __hash__(self):
        return hash((self.mlu_stream, self.device))

    def __repr__(self):
        return "<torch.mlu.Stream device={0} mlu_stream={1:#x}>".format(
            self.device, self.mlu_stream
        )


class ExternalStream(Stream):
    r"""Wrapper around an externally allocated MLU stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `cnrtQueue_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. if device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        with torch.mlu.device(device):
            return super().__new__(cls, stream_ptr=stream_ptr, **kwargs)


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
    streamdata = torch_mlu._MLUC._mlu_getCurrentMLUStream(
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


### Event
class Event(torch_mlu._MLUC._MLUEventBase):
    r"""Wrapper around a MLU event.

    MLU events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize MLU
    streams.

    The underlying MLU events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Arguments:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
    """

    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        return super(Event, cls).__new__(
            cls,
            enable_timing=enable_timing,
            blocking=blocking,
            interprocess=interprocess,
        )

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        return super(Event, cls).from_ipc_handle(device, handle)

    def record(self, stream=None):
        r"""Records the event in a given stream.

        Uses ``torch.mlu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device."""
        if stream is None:
            stream = torch.mlu.current_stream()
        super(Event, self).record(stream)

    def wait(self, stream=None):
        r"""Makes all future work submitted to the given stream wait for this
        event.

        Use ``torch.mlu.current_stream()`` if no stream is specified."""
        if stream is None:
            stream = torch.mlu.current_stream()
        super(Event, self).wait(stream)

    def query(self):
        r"""Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
        return super(Event, self).query()

    def elapsed_time(self, end_event):
        r"""Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.
        """
        return super(Event, self).elapsed_time(end_event)

    def hardware_time(self, end_event):
        time = super(Event, self).hardware_time(end_event)
        return time

    def synchronize(self):
        r"""Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``cnrtWaitNotifier``: see
            cnrt documentation`_ for more info.
        """
        super(Event, self).synchronize()

    def ipc_handle(self):  # pylint:disable=R0022
        r"""Returns an IPC handle of this event. If not recorded yet, the event
        will use the current device."""
        return super(Event, self).ipc_handle()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.mlu_event)

    def __repr__(self):
        if self.mlu_event:
            return "<torch.mlu.Event {0:#x}>".format(self._as_parameter_.value)
        else:
            return "<torch.mlu.Event uninitialized>"
