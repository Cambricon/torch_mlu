import ctypes
import torch
import torch_mlu
from torch.cuda._utils import _dummy_type


if not hasattr(torch_mlu._MLUC, "_MLUStreamBase"):
    # Define dummy base classes
    torch_mlu._MLUC.__dict__["_MLUStreamBase"] = _dummy_type("_MLUStreamBase")
    torch_mlu._MLUC.__dict__["_MLUEventBase"] = _dummy_type("_MLUEventBase")


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
