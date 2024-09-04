import os
import re
from typing import Union, Dict, Any, Tuple, Optional
import collections
import pickle
import sys
import warnings
from inspect import signature

import torch
from torch.cuda._memory_viz import segments as _segments, memory as _memory
import torch_mlu
from torch.types import Device
from ._utils import _get_device_index
from . import _lazy_init

__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "mem_get_info",
    "is_linear_memory_enabled",
    "enable_linear_memory",
]


def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    r"""Perform a memory allocation using the MLU memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.mlu.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MLU device is used.
        stream (torch.mlu.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.
    """
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = torch.mlu.current_stream(device)
    if isinstance(stream, torch.mlu.Stream):
        stream = stream.mlu_stream
    if not isinstance(stream, int):
        raise TypeError(
            "Invalid type for stream argument, must be "
            "`torch.mlu.Stream` or `int` representing a pointer "
            "to a exisiting stream"
        )
    with torch.mlu.device(device):
        return torch_mlu._MLUC._mlu_mluCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Delete memory allocated using the MLU memory allocator.

    Memory allocated with :func:`~torch.mlu.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.
    """
    torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete(mem_ptr)


def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = None
) -> None:
    r"""Set memory fraction for a process.

    The fraction is used to limit an caching allocator to allocated memory on a MLU device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default MLU device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")

    torch_mlu._MLUC._mlu_setMemoryFraction(fraction, device)


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other MLU application and visible in
    `cnmon info`.

    .. note::
        :func:`~torch.mlu.empty_cache` doesn't increase the amount of MLU
        memory available for PyTorch. However, it may help reduce fragmentation
        of MLU memory in certain cases.
    """
    if torch.mlu.is_initialized():
        torch_mlu._MLUC._mlu_emptyCache()


def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return a dictionary of MLU memory allocator statistics for a
    given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cnrtMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``cnrtMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the documentation).
    This helps avoid memory framentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cnrtMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Return the result of :func:`~torch.mlu.memory_stats` as a nested dictionary."""
    if not torch.mlu.is_initialized():
        _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_memoryStats(device)


def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the MLU memory allocator.

    See :func:`~torch.mlu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Reset the "peak" stats tracked by the MLU memory allocator.

    See :func:`~torch.mlu.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch_mlu._MLUC._mlu_resetPeakMemoryStats(device)


def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum MLU memory occupied by
    tensors for a given device.

    See :func:`~torch.mlu.max_memory_allocated` for details.

    .. warning::
        This function now calls :func:`~torch.mlu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    warnings.warn(
        "torch.mlu.reset_max_memory_allocated now calls torch.mlu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Reset the starting point in tracking maximum MLU memory managed by the
    caching allocator for a given device.

    See :func:`~torch.mlu.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    .. warning::
        This function now calls :func:`~torch.mlu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.
    """
    warnings.warn(
        "torch.mlu.reset_max_memory_cached now calls torch.mlu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Return the current MLU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Return the maximum MLU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.mlu.reset_peak_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Return the current MLU memory managed by the caching allocator in bytes
    for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Return the maximum MLU memory managed by the caching allocator in bytes
    for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.mlu.reset_peak_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


def memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.mlu.memory_reserved`."""
    warnings.warn(
        "torch.mlu.memory_cached has been renamed to torch.mlu.memory_reserved",
        FutureWarning,
    )
    return memory_reserved(device=device)


def max_memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.mlu.max_memory_reserved`."""
    warnings.warn(
        "torch.mlu.max_memory_cached has been renamed to torch.mlu.max_memory_reserved",
        FutureWarning,
    )
    return max_memory_reserved(device=device)


def memory_snapshot():
    r"""Return a snapshot of the mlu memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.
    """
    return torch_mlu._MLUC._mlu_memorySnapshot()["segments"]


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    r"""Returns a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "MLU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "MLU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch MLU memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} MLU OOMs: {num_ooms:<12d} | {_:6} cnrtMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                " {:<21} | {} | {} | {} | {} ".format(
                    submetric_name,
                    formatter(current, current_prefval),
                    formatter(peak, peak_prefval),
                    formatter(allocated, allocated_prefval),
                    formatter(freed, freed_prefval),
                ),
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize MLU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(
            " {:<21} | {} | {} | {} | {} ".format(
                metric_name,
                formatter(current, current),
                formatter(peak, peak),
                formatter(allocated, allocated),
                formatter(freed, freed),
            ),
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    r"""Return the global free and total MLU memory occupied for a given
    device using cnrtMemGetInfo.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mlu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if device is None:
        device = torch.mlu.current_device()
    device = _get_device_index(device)
    return torch_mlu._MLUC._mlu_mem_get_info(device)


linear_memory_enabled = None


def is_linear_memory_enabled():
    r"""
    Returns whether linear memory has enabled or not.
    """
    global linear_memory_enabled
    if linear_memory_enabled is not None:
        return linear_memory_enabled

    mlu_alloc_conf = os.environ.get("PYTORCH_MLU_ALLOC_CONF", "")
    match = re.search(r"use_linear_memory:([^,]*)", mlu_alloc_conf)
    if match:
        linear_memory_enabled = match.group(1) == "True"
    else:
        linear_memory_enabled = False
    return linear_memory_enabled


def enable_linear_memory():
    r"""
    Enable linear memory.
    """
    mlu_alloc_conf = os.environ.get("PYTORCH_MLU_ALLOC_CONF", "")
    if "use_linear_memory" not in mlu_alloc_conf:
        if mlu_alloc_conf:
            mlu_alloc_conf += ","
        os.environ["PYTORCH_MLU_ALLOC_CONF"] = mlu_alloc_conf + "use_linear_memory:True"
    else:
        if not is_linear_memory_enabled():
            warnings.warn(
                "Linear memory has already been disabled, which may cause performance degradation. "
                "You can enable it by setting PYTORCH_MLU_ALLOC_CONF=use_linear_memory:True"
            )


def _is_history_enabled():
    r"""
    Returns whether record history has enabled or not.
    """
    return torch_mlu._MLUC._mlu_isHistoryEnabled()


def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: Union[Device, int] = None,
    record_context_cpp=False,
):
    torch_mlu._MLUC._mlu_record_memory_history_legacy(
        enabled,
        record_context,
        trace_alloc_max_entries,
        trace_alloc_record_context,
        record_context_cpp,
    )


def _record_memory_history(enabled="all", *args, **kwargs):
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.mlu.memory._snapshot()`.

    In addition too keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.mlu.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currenly allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
    """
    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    else:
        return _record_memory_history_impl(enabled, *args, **kwargs)


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: Union[Device, int] = None,
):
    torch_mlu._MLUC._mlu_record_memory_history(enabled, context, stacks, max_entries)


_record_memory_history.__signature__ = signature(_record_memory_history_impl)


def _snapshot(device: Union[Device, int] = None):
    """Save a snapshot of MLU memory state at the time it was called.

    The state is represented as a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments : List[Segment]
            device_traces: List[List[TraceEntry]]

        class Segment(TypedDict):
            # Segments are memory returned from a cnrtMalloc call.
            # The size of reserved memory is the sum of all Segments.
            # Segments are cached and reused for future allocations.
            # If the reuse is smaller than the segment, the segment
            # is split into more then one Block.
            # empty_cache() frees Segments that are entirely inactive.
            address: int
            total_size: int #  cnrtMalloc'd size of segment
            stream: int
            segment_type: Literal['small', 'large'] # 'large' (>1MB)
            allocated_size: int # size of memory in use
            active_size: int # size of memory in use or in active_awaiting_free state
            blocks : List[Block]

        class Block(TypedDict):
            # A piece of memory returned from the allocator, or
            # current cached but inactive.
            size: int
            requested_size: int # size requested during malloc, may be smaller than
                                # size due to rounding
            address: int
            state: Literal['active_allocated', # used by a tensor
                        'active_awaiting_free', # waiting for another stream to finish using
                                                # this, then it will become free
                        'inactive',] # free for reuse
            frames: List[Frame] # stack trace from where the allocation occurred

        class Frame(TypedDict):
                filename: str
                line: int
                name: str

        class TraceEntry(TypedDict):
            # When `torch.mlu.memory._record_memory_history()` is enabled,
            # the snapshot will contain TraceEntry objects that record each
            # action the allocator took.
            action: Literal[
            'alloc'  # memory allocated
            'free_requested', # the allocated received a call to free memory
            'free_completed', # the memory that was requested to be freed is now
                            # able to be used in future allocation calls
            'segment_alloc', # the caching allocator ask cnrtMalloc for more memory
                            # and added it as a segment in its cache
            'segment_free',  # the caching allocator called cnrtFree to return memory
                            # to cnrt possibly trying free up memory to
                            # allocate more segments or because empty_caches was called
            'oom',          # the allocator threw an OOM exception. 'size' is
                            # the requested number of bytes that did not succeed
            'snapshot'      # the allocator generated a memory snapshot
                            # useful to coorelate a previously taken
                            # snapshot with this trace
            ]
            addr: int # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int # only present for OOM, the amount of
                            # memory cnrt still reports to be free

    Returns:
        The Snapshot dictionary object
    """
    return torch_mlu._MLUC._mlu_memorySnapshot()


def _dump_snapshot(filename="dump_snapshot.pickle"):
    """
    Save a pickled version of the `torch.mlu.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
    """
    s = _snapshot()
    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_segments(snapshot))


def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = memory_snapshot()
    with open(filename, "w") as f:
        f.write(_memory(snapshot))


def _set_allocator_settings(env: str):
    return torch_mlu._MLUC._mlu_mluCachingAllocator_set_allocator_settings(env)
