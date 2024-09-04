import copy
import functools
import operator
import logging
import math
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.autograd.profiler as autograd_profiler
from torch._inductor.triton_heuristics import CachingAutotuner
from torch._dynamo.utils import dynamo_timed, get_first_attr
from torch._inductor import config
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.utils import (
    do_bench,
    conditional_product,
    next_power_of_2,
    triton_config_to_hashable,
)
from torch._inductor.codecache import cache_dir
from torch._inductor.coordinate_descent_tuner import CoordescTuner
from torch_mlu._dynamo.device_interface import get_interface_for_device

from torch.utils._triton import has_triton_package
from torch._inductor.triton_heuristics import (
    HeuristicType,
    cached_autotune,
    autotune_hints_to_configs,
    disable_pointwise_autotuning,
    _NUM_THREADS_PER_WARP,
    triton_config_reduction,
)
from torch.mlu.memory import is_linear_memory_enabled

log = logging.getLogger(__name__)
maxGridSize = 65535

if has_triton_package():
    import triton
    from triton import Config
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.jit import KernelInterface

    try:
        from triton.compiler.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
    except ImportError:
        ASTSource = None
else:
    Config = object
    triton = None
    KernelInterface = object
    OutOfResources = object
    ASTSource = None


def check_config(cfg, *, xnumel=None, ynumel=None, znumel=None):
    for numel, label in zip((xnumel, ynumel, znumel), "XYZ"):
        if numel is None:
            continue
        block = cfg[f"{label}BLOCK"]
        if numel == 1:
            assert block == 1, (
                f"TritonKernel.indexing assumes numel == 1 => BLOCK == 1"
                f" but {label.lower()}numel=={numel} and {label}BLOCK={block} (cfg={cfg})."
            )
        max_block = config.triton.max_block[label]
        max_block_str = f'config.triton.max_block["{label}"]'
        if numel / block > maxGridSize:
            return False
    return True


def get_total_block_size(cfg):
    block = 1
    for label in "XYZ":
        key = f"{label}BLOCK_FRAGMENT"
        if key in cfg.kwargs:
            block *= cfg.kwargs[key]
    return block


def unique_configs(configs: List[Config]):
    """Remove duplicate configurations"""
    seen = set()
    pruned_configs = []

    for cfg in configs:
        if cfg is None:
            continue
        key = triton_config_to_hashable(cfg)
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs


torch._inductor.triton_heuristics.unique_configs = unique_configs


def __init__(
    self,
    fn,
    triton_meta,  # passed directly to triton
    configs,
    save_cache_hook,
    mutated_arg_names,
    heuristic_type,
    size_hints=None,
    inductor_meta=None,  # metadata not relevant to triton
    custom_kernel=False,  # whether the kernel is inductor-generated or custom
):
    super(CachingAutotuner, self).__init__()

    assert len(configs) > 0, "Non-empty TritonConfig list required for compiling"
    self.fn = fn
    self.triton_meta = triton_meta
    self.inductor_meta = {} if inductor_meta is None else inductor_meta
    self.save_cache_hook = save_cache_hook
    self.mutated_arg_names = mutated_arg_names
    self.configs = configs
    self.heuristic_type = heuristic_type
    self.custom_kernel = custom_kernel
    self.cuda_kernel_saved = False

    # Align the default design that default as mlu
    self.device_type = (
        triton_meta["device_type"] if "device_type" in triton_meta else "mlu"
    )
    self.gpu_device = get_interface_for_device(self.device_type)

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "CachingAutotuner gets %d configs for %s",
            len(self.configs),
            self.fn.__name__,
        )
        for c in self.configs:
            log.debug(c)

    self.launchers = []
    self.lock = threading.Lock()
    if os.getenv("TRITON_CACHE_DIR") is None:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(
            cache_dir(),
            "triton",
            str(self.triton_meta.get("device", 0)),
        )

    self.size_hints = size_hints
    self.coordesc_tuner = CoordescTuner(
        is_mm=False, name=self.fn.__name__, size_hints=size_hints
    )

    # pre-create the profiler context manager to reduce latency
    self.record_function_ctx = torch._C._profiler._RecordFunctionFast(
        self.inductor_meta.get("kernel_name", "triton kernel")
    )


torch._inductor.triton_heuristics.CachingAutotuner.__init__ = __init__


def bench(self, launcher, *args, grid, **kwargs):
    """Measure the performance of a given launcher"""
    # we don't skip configs wiht spilled registers when auto-tuning custom
    # (user-written) Triton kernels, as (i) we don't have any knowledge or
    # control over the kernel code; (ii) there is empirical evidence that
    # for some (complicated) custom Triton kernels, a register-spilling
    # config may yield the best latency.
    if not self.custom_kernel and launcher.n_spills > config.triton.spill_threshold:
        log.debug(
            "Skip config %s because of register spilling: %d",
            launcher.config,
            launcher.n_spills,
        )
        return float("inf")

    stream = torch.mlu.current_stream(torch.mlu.current_device()).mlu_stream

    def kernel_call():
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook(
                {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
            )

        cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
        launcher(
            *cloned_args,
            **cloned_kwargs,
            grid=grid,
            stream=stream,
        )

    return do_bench(kernel_call, rep=40, fast_flush=True)


torch._inductor.triton_heuristics.CachingAutotuner.bench = bench


def run(self, *args, grid, stream, **kwargs):
    if len(self.launchers) != 1:
        if len(self.launchers) == 0:
            self.precompile()
        if len(self.launchers) > 1:
            self.autotune_to_one_config(*args, grid=grid, **kwargs)

    if (
        not getattr(self.launchers[0].config, "found_by_coordesc", False)
        and config.coordinate_descent_tuning
    ):
        self.launchers = [
            self.coordinate_descent_tuning(
                self.launchers[0], *args, grid=grid, **kwargs
            )
        ]

    (launcher,) = self.launchers
    if launcher.store_cubin:
        self.save_cuda_kernel(grid, stream, launcher)

    if launcher.config.pre_hook is not None:
        launcher.config.pre_hook(
            {**dict(zip(self.arg_names, args)), **launcher.config.kwargs, **kwargs}
        )

    # guard the record_function_ctx and only call it if profiling is currently
    # in progress, to reduce latency when profiler is not turned on. Note that
    # the "if" statement (instead of, say, a contextlib.nullcontext) is intentional;
    # it is faster than entering and exiting a context manager, even if the context
    # manager is a nullcontext.
    if autograd_profiler._is_profiler_enabled:
        with self.record_function_ctx:
            return launcher(
                *args,
                **kwargs,
                grid=grid,
                stream=stream.mlu_stream,
            )
    else:
        return launcher(
            *args,
            **kwargs,
            grid=grid,
            stream=stream.mlu_stream,
        )


torch._inductor.triton_heuristics.CachingAutotuner.run = run


def precompile(self, warm_cache_only_with_cc=None):
    with self.lock:
        if self.launchers:
            return
        self.launchers = []
        compiled_binaries = []
        if not self.configs:
            raise RuntimeError("No triton configs are available")

        max_block_size = sys.maxsize
        for c in self.configs:
            # print("current config:", c)
            if not config.max_autotune and len(self.launchers) > 0:
                break

            total_block_size = get_total_block_size(c)
            if total_block_size >= max_block_size:
                break

            try:
                compiled_binary, launcher = self._precompile_config(
                    c, warm_cache_only_with_cc
                )
            except OutOfResources:
                # Skip the config if we run out of resource
                continue
            except RuntimeError as e:
                max_block_size = total_block_size
                continue
            self.launchers.append(launcher)
            compiled_binaries.append(compiled_binary)

        if len(self.launchers) == 0:
            raise RuntimeError(
                "No valid triton configs. Report a fatal compilation error"
            )

        seen_configs = set(self.configs)

        device_prop = self.gpu_device.Worker.get_device_properties(
            self.triton_meta["device"]
        )
        if (
            config.dynamic_scale_rblock
            and self.heuristic_type == HeuristicType.REDUCTION
            and self.size_hints is not None
            # Disable for AMDGPU as Triton is not ready to return n_regs for a compiled_binary.
            and torch.version.hip is None
            and device_prop.major >= 8
        ):
            for triton_config, compiled_binary in zip(self.configs, compiled_binaries):
                assert len(self.size_hints) == 2
                xblock = triton_config.kwargs.get("XBLOCK", 1)
                rblock = triton_config.kwargs["RBLOCK"]
                total_block = (self.size_hints[0] + xblock - 1) // xblock
                nreg = getattr(compiled_binary, "n_regs", None)
                if nreg is None:
                    continue

                # make sure rblock is not too small
                if rblock <= 64:
                    continue

                # each SM of A100 has 65536 32-bit registers. To maximize
                # the theoretical occupancy, we need run 2048 threads on each
                # SM. So each thread should use no more than 65536 / 2048
                # = 32 registers. In cases where occupancy matters, and each
                # thread uses too many registers, reduce RBLOCK to reduce
                # the register usage.
                # For kernel https://gist.github.com/shunting314/e4cccc031fe30d378b9b23c08c238cbd
                # from PLBartForCausalLM, latency improve from
                # 7.795ms to 4.883ms.
                #
                if (
                    nreg
                    <= device_prop.regs_per_multiprocessor
                    // device_prop.max_threads_per_multi_processor
                ):
                    continue

                nreg_per_warp = nreg * 32
                nreg_per_block = nreg_per_warp * triton_config.num_warps

                # Previously we set max_blocks_per_sm to 'max_threads_per_multi_processo / (32 * num_warps)'
                # The formula below is a tighter upper bound since we have the assumption that
                #   nreg > device_prop.regs_per_multiprocessor // device_prop.max_threads_per_multi_processor
                # due to the if condition above and:
                #   regs_per_multiprocessor / nreg_per_block
                #   = regs_per_multiprocessor / (nreg * 32 * num_warps)
                #   < regs_per_multiprocessor / ((regs_per_multiprocessor / max_threads_per_multi_processor) * 32 * num_warps)
                #   = max_threads_per_multi_processor / (32 * num_warps)
                # Using a tigher upper bound can reveal more optimization opportunities.
                max_blocks_per_sm = max(
                    device_prop.regs_per_multiprocessor // nreg_per_block, 1
                )

                if total_block <= max_blocks_per_sm * device_prop.multi_processor_count:
                    # no need to improve occupancy
                    continue
                new_config = copy.deepcopy(triton_config)
                new_config.kwargs["RBLOCK"] = rblock // 2
                if new_config in seen_configs:
                    continue
                seen_configs.add(new_config)
                self.launchers.append(
                    self._precompile_config(new_config, warm_cache_only_with_cc)[1]
                )
        self.configs = None


torch._inductor.triton_heuristics.CachingAutotuner.precompile = precompile


def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Optional[int]):
    """Ahead of time compile a given autotuner config."""
    compile_meta = copy.deepcopy(self.triton_meta)
    for k, v in cfg.kwargs.items():
        compile_meta["constants"][self.fn.arg_names.index(k)] = v
    compile_meta["num_warps"] = cfg.num_warps
    compile_meta["num_stages"] = cfg.num_stages
    compile_meta["silence"] = True
    compile_meta["debug"] = (
        config.assert_indirect_indexing and torch.version.hip is None
    )

    # print("compile kernel:", self.fn)
    # Setting device_type="hip" required on ROCm to pass down to triton
    compile_meta["device_type"] = (
        self.device_type if torch.version.hip is None else "hip"
    )

    if warm_cache_only_with_cc:
        cc = warm_cache_only_with_cc
    else:
        # Use device_type 'cuda' for both cuda and hip devices to retrieve
        # the compute capability.
        device_type = self.device_type if torch.version.hip is None else "mlu"
        device_id = compile_meta["device"]
        device = torch.device(device_type, device_id)
        cc = self.gpu_device.get_compute_capability(device)

    device_prop = self.gpu_device.Worker.get_device_properties(
        self.triton_meta["device"]
    )
    compile_meta["is_linear"] = is_linear_memory_enabled() if device_prop.supports_linear_memory else False
    compile_meta["cc"] = cc
    compile_meta["isa_version"] = cc

    if ASTSource:
        compile_args = (
            ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
                compile_meta["configs"][0],
            ),
        )

        target = GPUTarget(compile_meta["device_type"], cc, 1)
        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"],
            "silence": compile_meta["silence"],
            "isa_version": compile_meta["isa_version"],
            "is_linear": compile_meta["is_linear"],
        }
        compile_kwargs = {
            "target": target,
            "options": options,
        }
    else:
        compile_args = (self.fn,)
        compile_kwargs = compile_meta

    if warm_cache_only_with_cc:
        return (
            triton.compile(*compile_args, **compile_kwargs),
            None,
        )

    # load binary to the correct device
    with self.gpu_device.device(compile_meta["device"]):  # type: ignore[attr-defined]
        # need to initialize context
        self.gpu_device.synchronize(self.gpu_device.current_device())

        try:
            binary = triton.compile(*compile_args, **compile_kwargs)
        except RuntimeError:
            raise
        except Exception:
            log.exception(
                "Triton compilation failed: %s\n%s\nmetadata: %s",
                self.inductor_meta.get("kernel_name", "triton_"),
                self.fn.src,
                compile_meta,
            )
            raise
        binary._init_handles()

    call_args = [
        arg for i, arg in enumerate(self.fn.arg_names) if i not in self.fn.constexprs
    ]
    def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]

    binary_shared = (
        binary.shared if hasattr(binary, "shared") else binary.metadata.shared
    )

    scope = {
        "grid_meta": cfg.kwargs,
        "bin": binary,
        "launch_enter_hook": binary.launch_enter_hook,
        "launch_exit_hook": binary.launch_exit_hook,
        "metadata": binary.packed_metadata,
        # "metadata": binary.metadata,
        "shared": binary_shared,
    }

    scope["num_warps"] = (
        binary.num_warps if hasattr(binary, "num_warps") else binary.metadata.num_warps
    )

    scope["cta_args"] = (
        (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
        if hasattr(binary, "num_ctas")
        else (
            (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
            if hasattr(binary, "metadata")
            else ()
        )
    )

    scope["function"] = get_first_attr(binary, "function", "cu_function")

    def get_launch_args_without_kernel_launch_metadata(
        grid,
        grid_0,
        grid_1,
        grid_2,
        stream,
        function,
        metadata,
        bin,
        launch_enter_hook,
        launch_exit_hook,
        num_warps,
        shared,
        cta_args,
        args,
    ):
        """
        Construct launch args before CompiledKernel.launch_metadata is added.
        """
        return (
            grid_0,
            grid_1,
            grid_2,
            num_warps,
            *cta_args,
            shared,
            stream,
            function,
            launch_enter_hook,
            launch_exit_hook,
            metadata,
        )

    def get_launch_args_with_kernel_launch_metadata(
        grid,
        grid_0,
        grid_1,
        grid_2,
        stream,
        function,
        metadata,
        bin,
        launch_enter_hook,
        launch_exit_hook,
        num_warps,
        shared,
        cta_args,
        args,
    ):
        """
        Construct launch args after CompiledKernel.launch_metadata is added
        by https://github.com/openai/triton/pull/3492 .
        """
        return (
            grid_0,
            grid_1,
            grid_2,
            stream,
            function,
            metadata,
            bin.launch_metadata(grid, stream, *args),
            launch_enter_hook,
            launch_exit_hook,
        )

    scope["get_launch_args"] = (
        get_launch_args_with_kernel_launch_metadata
        if hasattr(binary, "launch_metadata")
        else get_launch_args_without_kernel_launch_metadata
    )

    scope["runner"] = get_first_attr(binary, "run", "c_wrapper")

    exec(
        f"""
        def launcher({', '.join(def_args)}, grid, stream):
            if callable(grid):
                grid_0, grid_1, grid_2 = grid(grid_meta)
            else:
                grid_0, grid_1, grid_2 = grid

            #print("  BLOCKS:", grid_meta, ", grids:", grid_0, grid_1, grid_2)
            args = {', '.join(call_args)},
            launch_args = get_launch_args(
                grid, grid_0, grid_1, grid_2, stream, function,
                metadata, bin, launch_enter_hook, launch_exit_hook,
                num_warps, shared, cta_args, args
            )
            runner(*launch_args, *args)
            return bin
        """.lstrip(),
        scope,
    )

    launcher = scope["launcher"]
    launcher.config = cfg
    launcher.n_regs = getattr(binary, "n_regs", None)
    launcher.n_spills = getattr(binary, "n_spills", None)
    launcher.shared = binary_shared
    launcher.store_cubin = config.triton.store_cubin
    # store this global variable to avoid the high overhead of reading it when calling run
    if launcher.store_cubin:
        launcher.fn = self.fn
        launcher.bin = binary

    return binary, launcher


torch._inductor.triton_heuristics.CachingAutotuner._precompile_config = (
    _precompile_config
)


@functools.lru_cache(None)
def all_candidate_steps(x, y=None, z=None):
    def candidate_steps(n):
        if n is None:
            return []
        if n <= 32:
            heuristics_list = list(range(2, math.ceil(n / 2), 2)) + [n]
        elif n <= 64:
            heuristics_list = list(range(4, math.ceil(n / 2), 4)) + [n]
        elif n <= 128:
            heuristics_list = list(range(8, math.ceil(n / 2), 8)) + [n]
        elif n <= 512:
            heuristics_list = list(range(64, math.ceil(n / 2), 64)) + [16, 32, n]
        elif n <= 1024:
            heuristics_list = list(range(128, math.ceil(n / 2), 128)) + [16, 32, 64, n]
        elif n <= 16384:
            heuristics_list = [n, 32, 256, 512] + list(
                range(1024, math.ceil(n / 2), 1024)
            )
        elif n <= 65536:
            heuristics_list = [n, 128, 1024, 4096] + list(
                range(8192, math.ceil(n / 2), 2048)
            )
        else:
            heuristics_list = [128, 1024, 8192, 12288] + list(range(16384, 65536, 8192))
        heuristics_list = sorted(heuristics_list, reverse=True)
        result_list = []
        for candidate in heuristics_list:
            remainder = n % candidate
            pad_num = candidate - remainder
            if remainder and pad_num / x > 0.5 and len(result_list) > 2:
                continue
            else:
                result_list.append(candidate)
        return result_list

    all_candidates = []
    x_candidates = candidate_steps(x)
    y_candidates = candidate_steps(y)
    z_candidates = candidate_steps(z)

    minimal_length = 1024
    max_length = 1048576

    if y is None and z is None:
        for x_candidate in x_candidates:
            if x_candidate >= max_length:
                continue
            all_candidates.append([x_candidate])
    elif y and z is None:
        for x_candidate in x_candidates:
            for y_candidate in y_candidates:
                length = conditional_product(x_candidate, y_candidate)
                if length >= max_length:
                    continue
                if len(all_candidates) > 1 and length < minimal_length:
                    continue
                all_candidates.append([y_candidate, x_candidate])
    elif y and z:
        for x_candidate in x_candidates:
            for y_candidate in y_candidates:
                for z_candidate in z_candidates:
                    length = conditional_product(x_candidate, y_candidate, z_candidate)
                    if length >= max_length:
                        continue
                    if len(all_candidates) > 1 and length < minimal_length:
                        continue
                    all_candidates.append([z_candidate, y_candidate, x_candidate])

    return all_candidates


def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    assert not inductor_meta.get("no_x_dim")
    core_num = 48
    num_warps = 1

    if len(size_hints) == 1:
        triton_configs = []
        blocksize = math.ceil(size_hints[0] / core_num)
        for step in all_candidate_steps(blocksize):
            cfg = {"XBLOCK": blocksize, "XBLOCK_FRAGMENT": step[0]}
            triton_configs.append(Config(cfg, num_warps=num_warps, num_stages=3))
        triton_configs = sorted(
            triton_configs, key=lambda cfg: get_total_block_size(cfg)
        )

        if disable_pointwise_autotuning() and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            triton_configs = [triton_configs[0]]

        return cached_autotune(
            size_hints,
            triton_configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.POINTWISE,
            filename=filename,
        )
    if len(size_hints) == 2:
        blocksizes = []
        remain_grid = 1
        if size_hints[0] <= core_num:
            blocksizes.append(1)
            remain_grid = core_num // size_hints[0]
        else:
            blocksizes.append(math.ceil(size_hints[0] / core_num))
        blocksizes.append(math.ceil(size_hints[1] / remain_grid))
        triton_configs = []
        for step in all_candidate_steps(blocksizes[1], blocksizes[0]):
            cfg = {
                "XBLOCK": blocksizes[1],
                "XBLOCK_FRAGMENT": step[1],
                "YBLOCK": blocksizes[0],
                "YBLOCK_FRAGMENT": step[0],
            }
            triton_configs.append(Config(cfg, num_warps=num_warps, num_stages=3))
        triton_configs = sorted(
            triton_configs, key=lambda cfg: get_total_block_size(cfg)
        )

        if (disable_pointwise_autotuning() or tile_hint == TileHint.SQUARE) and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            triton_configs = [triton_configs[0]]

        return cached_autotune(
            size_hints,
            triton_configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    if len(size_hints) == 3:
        blocksizes = []
        remain_grid = 1
        if size_hints[0] <= core_num:
            blocksizes.append(1)
            remain_grid = core_num // size_hints[0]
        else:
            blocksizes.append(math.ceil(size_hints[0] / core_num))
        if size_hints[1] <= remain_grid:
            blocksizes.append(1)
            remain_grid = remain_grid // size_hints[1]
        else:
            blocksizes.append(math.ceil(size_hints[1] / remain_grid))
            remain_grid = 1
        blocksizes.append(math.ceil(size_hints[2] / remain_grid))

        triton_configs = []
        for step in all_candidate_steps(blocksizes[2], blocksizes[1], blocksizes[0]):
            cfg = {
                "XBLOCK": blocksizes[2],
                "XBLOCK_FRAGMENT": step[2],
                "YBLOCK": blocksizes[1],
                "YBLOCK_FRAGMENT": step[1],
                "ZBLOCK": blocksizes[0],
                "ZBLOCK_FRAGMENT": step[0],
            }
            triton_configs.append(Config(cfg, num_warps=num_warps, num_stages=3))

        triton_configs = sorted(
            triton_configs, key=lambda cfg: get_total_block_size(cfg)
        )
        if disable_pointwise_autotuning():
            triton_configs = [triton_configs[0]]

        return cached_autotune(
            size_hints,
            triton_configs,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


torch._inductor.triton_heuristics.pointwise = pointwise


def _reduction_configs(
    *, size_hints: List[int], inductor_meta: Dict[str, Any]
) -> List[Config]:
    reduction_hint = inductor_meta.get("reduction_hint", None)
    assert len(size_hints) == 2
    rnumel = size_hints[-1]

    contiguous_config = triton_config_reduction(
        size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048)
    )
    outer_config = triton_config_reduction(size_hints, 64, 8)
    tiny_config = triton_config_reduction(
        size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048)
    )
    if config.max_autotune or config.max_autotune_pointwise:
        pass  # skip all these cases
    elif reduction_hint == ReductionHint.INNER:
        return [contiguous_config]
    elif reduction_hint == ReductionHint.OUTER:
        return [outer_config]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        return [tiny_config]
    if disable_pointwise_autotuning():
        return [triton_config_reduction(size_hints, 32, 128)]

    triton_configs = [contiguous_config, outer_config, tiny_config]
    for x_block in [16, 32]:
        for r_block in [32, 64, 256, 512, 1024, 4096, 8192]:
            if x_block <= size_hints[0] and r_block <= size_hints[1]:
                triton_configs.append(
                    triton_config_reduction(size_hints, x_block, r_block)
                )

    return triton_configs


torch._inductor.triton_heuristics._reduction_configs = _reduction_configs


def triton_config_reduction(
    size_hints, x, r, num_stages=0, num_warps=None
) -> Config:  # NOSONAR
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """

    target = conditional_product(x, r)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    r = min(r, size_hints[1])

    # if we are below original block size, scale up where we can
    while x < size_hints[0] and conditional_product(x, r) < target:
        x *= 2
    while r < size_hints[1] and conditional_product(x, r) < target:
        r *= 2

    cfg = {"XBLOCK": x, "RBLOCK": r}
    num_warps = 1
    if not check_config(cfg, xnumel=size_hints[0]):
        return None
    assert (
        r <= config.triton.max_block["R"]
    ), f"increase config.triton.MAX_BLOCK['r'] to {r}"
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


torch._inductor.triton_heuristics.triton_config_reduction = triton_config_reduction
