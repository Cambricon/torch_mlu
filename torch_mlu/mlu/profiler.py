# mypy: allow-untyped-defs
import contextlib

from . import cnrt, check_error

__all__ = ["init", "start", "stop", "profile"]


def start():
    r"""Starts mlu profiler data collection.

    .. warning::
        Raises cnrtError in case of it is unable to start the profiler.
    """
    check_error(cnrt().mluProfilerStart())


def stop():
    r"""Stops mlu profiler data collection.

    .. warning::
        Raises cnrtError in case of it is unable to stop the profiler.
    """
    check_error(cnrt().mluProfilerStop())


def init():
    raise RuntimeError("Do not need to call init function to start MLU profiler.")


@contextlib.contextmanager
def profile():
    """
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from MLU backend.
    Example:
        >>> import torch;import torch_mlu
        >>> model = torch.nn.Linear(20, 30).mlu()
        >>> inputs = torch.randn(128, 20).mlu()
        >>> with torch.mlu.profiler.profile() as prof:
        ...     model(inputs)

    Needs to be used in conjunction with command "cnperf-cli record --capture_range=cnProfilerApi".
    """
    try:
        start()
        yield
    finally:
        stop()
