import logging
import torch
from torch._inductor.virtualized import V
from torch._inductor import config
from torch._inductor.exc import CUDACompileError
from torch._inductor.select_algorithm import (
    VERIFY,
    DEBUG,
    ExternKernelCaller,
    AlgorithmSelectorCache,
    ErrorFromChoice,
)

log = logging.getLogger(__name__)


@classmethod
def make_benchmark_fn(
    cls,
    choices,
    input_nodes,
    layout,
    input_gen_fns=None,
):
    if input_gen_fns is None:
        input_gen_fns = {}

    # de-duplicate args
    unique_example_inputs = {
        x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x)
        for i, x in enumerate(input_nodes)
    }
    example_inputs = list(unique_example_inputs.values())
    example_inputs_extern = [
        torch.as_strided(
            unique_example_inputs[input_node.get_name()],
            V.graph.sizevars.size_hints(
                input_node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            V.graph.sizevars.size_hints(
                input_node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            V.graph.sizevars.size_hint(
                input_node.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            ),
        )
        for input_node in input_nodes
    ]

    out = cls.benchmark_example_value(layout)
    out_extern = torch.as_strided(
        out, out.size(), out.stride(), V.graph.sizevars.size_hint(layout.offset)
    )
    if VERIFY:
        choices[0].benchmark(*example_inputs_extern, out=out_extern)
        expected = out_extern.clone()

    if DEBUG:
        print(f"{len(choices)} tuning requests:")

    def debug_str():
        def tensor_repr(x):
            return (
                f"torch.empty_strided({tuple(x.size())!r}, {tuple(x.stride())!r}, "
                f"dtype={x.dtype!r}, device={x.device.type!r})"
            )

        lines = [
            "inputs = [",
        ]
        for x in example_inputs:
            lines.append(f"    {tensor_repr(x)},")
        lines += ["]", f"out = {tensor_repr(out)}", ""]
        return "\n".join(lines)

    def benchmark_choice_in_current_process(choice):
        out.zero_()
        if isinstance(choice, ExternKernelCaller):
            # aten kernels want the offset baked in for sliced tensors
            result = choice.benchmark(*example_inputs_extern, out=out_extern)
        else:
            # triton templates want the base pointer for sliced tensors
            result = choice.benchmark(*example_inputs, out=out)
        if VERIFY:
            torch.testing.assert_close(out_extern, expected, **VERIFY)
        # torch.cuda.synchronize()  # shake out any CUDA errors
        torch.mlu.synchronize()  # shake out any CUDA errors
        return result

    def benchmark_in_current_process(choices):
        timings = {}
        for choice in choices:
            try:
                timing = benchmark_choice_in_current_process(choice)
            except CUDACompileError as e:
                log.warning(
                    "MLU compilation error: \n%s. \nIgnore this choice.", str(e)
                )
                timing = float("inf")
            except RuntimeError as e:
                msg = str(e)
                if "invalid argument" in msg:
                    msg += "\n\nThis may mean this GPU is too small for max_autotune mode.\n\n"
                    log.warning(msg)
                    timing = float("inf")
                else:
                    if "illegal memory access" in msg:
                        msg += "\n\nEither error in template or triton bug.\n"
                    # raise ErrorFromChoice(msg, choice, debug_str())  # noqa: TRY200
                    # log.warning(msg)
                    timing = float("inf")
            except AssertionError as e:
                raise AssertionError(  # noqa: TRY200
                    f"Incorrect result from choice {choice}\n\n{e}"
                )

            timings[choice] = timing

        return timings

    def benchmark_in_sub_process(choices):
        from . import autotune_process

        # only benchmark triton kernel in sub process for now.
        # ATen/Extern kernel are still benchmarked in the current process.
        extern = [c for c in choices if isinstance(c, ExternKernelCaller)]
        triton = [c for c in choices if not isinstance(c, ExternKernelCaller)]

        timings = benchmark_in_current_process(extern)
        timings.update(autotune_process.benchmark_in_sub_process(triton))
        return timings

    benchmark = (
        benchmark_in_sub_process
        if config.autotune_in_subproc
        else benchmark_in_current_process
    )

    return benchmark


torch._inductor.select_algorithm.AlgorithmSelectorCache.make_benchmark_fn = (
    make_benchmark_fn
)
torch._inductor.select_algorithm.PRINT_AUTOTUNE = False
