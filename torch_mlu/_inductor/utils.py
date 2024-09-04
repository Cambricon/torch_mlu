import functools
import logging
from typing import (
    Any,
    Callable,
    List,
)
import torch
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch._inductor import utils
from torch._inductor.utils import use_max_autotune, timed

log = logging.getLogger(__name__)


@functools.lru_cache(None)
def is_big_gpu(index):
    return True
    # sms = torch.mlu.get_device_properties(index).multi_processor_count
    # if sms < 80:  # V100
    #     log.warning("not enough SMs to use max_autotune_gemm mode")
    #     return False
    # return True


torch._inductor.utils.is_big_gpu = is_big_gpu


def _use_template_for_cuda(layout, allowed_layout_dtypes: List[torch.dtype]) -> bool:
    return (
        use_max_autotune()
        and layout.device.type == "mlu"
        and layout.dtype in allowed_layout_dtypes
        and is_big_gpu(layout.device.index or 0)
    )


torch._inductor.utils._use_template_for_cuda = _use_template_for_cuda


def do_bench_using_profiling(fn: Callable[[], Any], warmup=25, rep=100) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    fn()
    torch.mlu.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="mlu")

    # Estimate the runtime of the function
    start_event = torch.mlu.Event(enable_timing=True)
    end_event = torch.mlu.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.mlu.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.MLU,
        ]
    ) as p:
        # Benchmark
        for i in range(n_repeat):
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            fn()
        # Record clocks
        torch.mlu.synchronize()

    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_mlu_time_total", row_limit=-1))

    filtered_events = EventList(
        [
            event
            for event in p.events()
            if event.device_type == DeviceType.PrivateUse1
            and event.name != "Context Sync"
        ]
    )
    if len(filtered_events) % n_repeat != 0:
        raise RuntimeError(
            "Failed to divide all profiling events into #repeat groups. "
            "#MLU events: %d, #repeats: %s",
            len(filtered_events),
            n_repeat,
        )
    num_event_per_group = len(filtered_events) / n_repeat
    actual_events = EventList(
        [
            event
            for i, event in enumerate(filtered_events)
            if i % num_event_per_group != 0
        ]
    )
    actual_events._build_tree()
    actual_events = actual_events.key_averages()

    log.debug("profiling time breakdown")
    log.debug(actual_events.table(row_limit=-1))

    res = sum(event.mlu_time_total for event in actual_events) / 1000.0 / n_repeat
    log.debug("profiling results: %s ms", res)
    return res


torch._inductor.utils.do_bench_using_profiling = do_bench_using_profiling


def print_performance(
    fn, args=(), times=10, repeat=10, baseline=1.0, device: str = "mlu"
):
    timings = torch.tensor([timed(fn, args, times, device) for _ in range(repeat)])
    took = torch.median(timings) / times
    print(f"{took/baseline:.6f}")
    return took


torch._inductor.utils.print_performance = print_performance
