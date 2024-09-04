from contextlib import contextmanager

try:
    from torch_mlu._MLUC import _cnpx
except ImportError:

    class _CNPXStub(object):
        @staticmethod
        def _fail(*args, **kwargs):
            raise RuntimeError(
                "CNPX functions not installed. Are you sure you have a torch_mlu build?"
            )

        rangePushA = _fail
        rangePop = _fail
        markA = _fail

    _cnpx = _CNPXStub()  # type: ignore[assignment]

__all__ = ["range_push", "range_pop", "range_start", "range_end", "mark", "range"]


def range_push(msg):
    """
    Pushes a range onto a stack of nested range span.

    Args:
        msg (str): ASCII message to associate with range

    Return:
        int. If an error occurs, a negative value is returned
    """
    return _cnpx.rangePush(msg)


def range_pop():
    """
    Pops a range off of a stack of nested range spans.

    Return:
        int. If an error occurs, a negative value is returned
    """
    return _cnpx.rangePop()


def range_start(msg):
    """
    Mark the start of a range with string message. It returns an unique handle
    for this range to pass to the corresponding call to range_end().

    A key difference between this and range_push/range_pop is that the
    range_start/range_end version supports range across threads (start on one
    thread and end on another thread).

    Returns:
        A range handle that can be passed to range_end().

    Args:
        msg (str): ASCII message to associate with the range.
    """
    return _cnpx.rangeStart(msg)


def range_end(range_id) -> None:
    """
    Mark the end of a range for a given range_id.

    Args:
        range_id (int): an unique handle for the start range.
    """
    _cnpx.rangeEnd(range_id)


def mark(msg):
    """
    Describe an instantaneous event that occurred at some point.

    Args:
        msg (str): ASCII message to associate with the event.
    """
    _cnpx.mark(msg)


@contextmanager
def range(msg, *args, **kwargs):
    """
    Context manager / decorator that pushes an CNPX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    range_push(msg.format(*args, **kwargs))
    yield
    range_pop()
