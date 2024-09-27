import os

import torch
import torch_mlu

from typing import Optional, Dict
from torch._C._profiler import _ExperimentalConfig

from torch.autograd import (
    _disable_profiler,
    _enable_profiler,
    ProfilerConfig,
    ProfilerState,
    DeviceType,
)

from .analysis.api import analyze_data


# A global dict (external id -> tuple(shapes, dtypes)) used for
# saving CPU-op's info, so that we can lookup it (of capturing
# stage) in mlu graph replay stage.
_id2opinfo: Dict[int, tuple] = {}


def _mlu_graph_saver_fn(prof):
    # only save info when record_shapes=True and enabled catching mlu graph
    if prof.record_shapes and os.environ.get(
        "TORCH_MLU_ENABLE_CATCHING_MLUGRAPH_OP", "FALSE"
    ).upper() in ["1", "TRUE", "ON"]:
        for evt in prof.profiler.kineto_results.events():
            # Since we cannot get evt.activityType(), we need to handle runtime case
            # that id > 0 but linked_id = 0, e.g.,cnTaskTopoEntityInvoke, cnrtSyncDevice.
            # At present, filter these non-cpu-op cases by using shapes and dtypes.
            if (
                evt.device_type() != DeviceType.CPU
                or evt.linked_correlation_id() > 0
                or (evt.shapes() == [] and evt.dtypes() == [])
            ):
                continue
            op_id = evt.correlation_id()
            if op_id != 0:
                assert (
                    op_id not in _id2opinfo
                ), f"CPU event with id {op_id} has been in dict!"
                _id2opinfo[op_id] = (evt.shapes(), evt.dtypes())


def insert_hook_for_profiler():
    """
    hijack _trace_ready method of torch.profiler.profiler.profiler
    and insert a pre hook to save cpu-op event's info in global dict.
    """

    def _hijack_method_of_cls(cls, method_name):
        orig_method = getattr(cls, method_name)

        def wrapped_method(*args, **kwargs):
            _mlu_graph_saver_fn(*args, **kwargs)
            return orig_method(*args, **kwargs)

        setattr(cls, method_name, wrapped_method)

    _hijack_method_of_cls(torch.profiler.profiler.profile, "_trace_ready")


def tensorboard_trace_handler(
    dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False
):
    import os
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(os.path.join(dir_name, file_name))
        analyze_data(os.path.join(dir_name, file_name), _id2opinfo)

    return handler_fn


class emit_cnpx(object):
    """Context manager that makes every autograd operation emit an CNPX range.

    It is useful when running the program under cnperf-cli

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args:
        enabled (bool, optional): Setting ``enabled=False`` makes this context manager a no-op.
            Default: ``True``.
        record_shapes (bool, optional): If ``record_shapes=True``, the cnpx range wrapping
            each autograd op will append information about the sizes of Tensor arguments received
            by that op, in the following format:
            ``[[arg0.size(0), arg0.size(1), ...], [arg1.size(0), arg1.size(1), ...], ...]``
            Non-tensor arguments will be represented by ``[]``.
            Arguments will be listed in the order they are received by the backend op.
            Please note that this order may not match the order in which those arguments were passed
            on the Python side.  Also note that shape recording may increase the overhead of cnpx range creation.
            Default: ``False``

    Example:
        >>> with torch.autograd.profiler.emit_cnpx():
        ...     model(x)

    **Forward-backward correlation**

    When viewing a profile created using :class:`emit_cnpx` in the chrome://tracing/,
    correlating each backward-pass op with the corresponding forward-pass op can be difficult.
    To ease this task, :class:`emit_cnpx` appends sequence number information to the ranges it
    generates.

    During the forward pass, each function range is decorated with ``seq=<N>``.  ``seq`` is a running
    counter, incremented each time a new backward Function object is created and stashed for backward.
    Thus, the ``seq=<N>`` annotation associated with each forward function range tells you that
    if a backward Function object is created by this forward function,
    the backward object will receive sequence number N.
    During the backward pass, the top-level range wrapping each C++ backward Function's
    ``apply()`` call is decorated with ``stashed seq=<M>``.  ``M`` is the sequence number that
    the backward object was created with.  By comparing ``stashed seq`` numbers in backward with ``seq``
    numbers in forward, you can track down which forward op created each backward Function.

    Any functions executed during the backward pass are also decorated with ``seq=<N>``.  During
    default backward (with ``create_graph=False``) this information is irrelevant, and in fact,
    ``N`` may simply be 0 for all such functions.  Only the top-level ranges associated with
    backward Function objects' ``apply()`` methods are useful, as a way to correlate these Function
    objects with the earlier forward pass.

    **Double-backward**

    If, on the other hand, a backward pass with ``create_graph=True`` is underway (in other words,
    if you are setting up for a double-backward), each function's execution during backward
    is given a nonzero, useful ``seq=<N>``.  Those functions may themselves create Function objects
    to be executed later during double-backward, just as the original functions in the forward pass did.
    The relationship between backward and double-backward is conceptually the same as the relationship
    between forward and backward: The functions still emit current-sequence-number-tagged ranges,
    the Function objects they create still stash those sequence numbers, and during the eventual
    double-backward, the Function objects' ``apply()`` ranges are still tagged with ``stashed seq``
    numbers, which can be compared to `seq` numbers from the backward pass.

    .. warning:
        The sequence number is thread-local, and some forward functions don't create an associated
        backward Function object (instead delegating that to sub-functions further down the call chain).
        For these reasons, the correspondence of stashed sequence numbers in
        backward Function ``apply()`` ranges with `seq` numbers in forward-pass ranges is
        not guaranteed to be 1 to 1.  The sequence numbers alone may not be enough to fully
        disambiguate which forward function created which
        backward Function object.  You may need to make a judgment based on analytic knowledge of what
        the expected correspondence should be.
    """

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("CNPX annotation context manager is not reentrant")
        self.entered = True
        torch.mlu.synchronize()
        # _run_on_profiler_start()
        # TODO: Registers a hook within dynamo eval_frame.c called before and after
        # the lookup process
        _enable_profiler(
            ProfilerConfig(
                ProfilerState.CNPX,
                self.record_shapes,
                False,
                False,
                False,
                False,
                _ExperimentalConfig(),
            ),
            set(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        torch.mlu.synchronize()
        _disable_profiler()
        # _run_on_profiler_stop()
        return False
