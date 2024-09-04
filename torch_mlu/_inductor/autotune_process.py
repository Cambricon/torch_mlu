import time
import functools
import logging
from typing import Optional, Callable

import torch
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    TensorMeta,
    TritonBenchmarkRequest,
)
from torch._inductor.utils import do_bench
from torch._inductor.codecache import PyCodeCache

log = logging.getLogger(__name__)
MLU_VISIBLE_DEVICES = "MLU_VISIBLE_DEVICES"


def benchmark(
    self,
    *input_tensors: torch.Tensor,
    output_tensor: Optional[torch.Tensor] = None,
) -> float:
    debug = log.isEnabledFor(logging.DEBUG)
    if debug:
        start_ts = time.time()

    # create args and out tensor
    if output_tensor is None:
        assert len(input_tensors) == 0
        input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
        output_tensor = self.output_tensor_meta.to_tensor()

    if debug:
        create_tensor_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
        start_ts = time.time()

    fn = self.make_run_fn(*input_tensors, output_tensor=output_tensor)

    if debug:
        load_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
        start_ts = time.time()

    # xdim must be less than 65536
    if self.grid[0] > 65535:
        return float("inf")
    out = do_bench(fn)
    torch.mlu.synchronize()  # shake out any CUDA errors

    if debug:
        bench_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
        log.debug(
            "InChildProcess %s: load %f, create tensor %f, bench %f",
            str(self),
            load_elapse,  # type: ignore[possibly-undefined]
            create_tensor_elapse,  # type: ignore[possibly-undefined]
            bench_elapse,
        )
    self.cleanup_run_fn()
    return out


torch._inductor.autotune_process.BenchmarkRequest.benchmark = benchmark


def make_run_fn(
    self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
) -> Callable[[], None]:
    mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
    log.debug(
        "benchmark module key: %s, path: %s",
        self.module_cache_key,
        self.module_path,
    )

    run_method = getattr(mod, self.kernel_name).run
    extra_args = list(self.extra_args)

    # Newer version of triton add warmup argument to JITFunction.run.
    # This code handles backward-compatibility.
    warmup_arg = {}
    import inspect

    if "warmup" in inspect.signature(run_method).parameters:
        warmup_arg["warmup"] = False

    if torch.version.hip and self.matrix_instr_nonkdim != 0:
        return functools.partial(
            run_method,
            *input_tensors,
            output_tensor,
            *self.extra_args,
            grid=self.grid,
            **warmup_arg,
            num_stages=self.num_stages,
            num_warps=self.num_warps,
            matrix_instr_nonkdim=self.matrix_instr_nonkdim,
        )
    else:
        return functools.partial(
            run_method,
            *input_tensors,
            output_tensor,
            *self.extra_args,
            grid=self.grid,
            **warmup_arg,
            num_stages=self.num_stages,
            num_warps=self.num_warps,
            # Modified by Cambricon start: add one line
            silence=True,
            # Modified by Cambricon end
        )


torch._inductor.autotune_process.TritonBenchmarkRequest.make_run_fn = make_run_fn
