import functools
import itertools
import logging
import sympy
from io import StringIO
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch

import torch
from torch._dynamo.utils import identity
from torch._inductor import ir
from torch._inductor.autotune_process import TensorMeta, TritonBenchmarkRequest
from torch._inductor.utils import sympy_product, unique, Placeholder
from torch._inductor.virtualized import V
from torch._inductor import config
from torch._inductor.exc import CUDACompileError
from torch._inductor.codecache import PyCodeCache
from torch._inductor.codegen.common import PrimitiveInfoType
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.select_algorithm import (
    VERIFY,
    DEBUG,
    ExternKernelCaller,
    AlgorithmSelectorCache,
    TritonTemplateCaller,
    TritonTemplateKernel,
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


def generate(
    self,
    input_nodes,
    layout,
    num_stages,
    num_warps,
    output_node=None,
    prefix_args=0,
    suffix_args=0,
    epilogue_fn=identity,
    mutated_inputs=None,
    **kwargs,
):
    assert self.template, "requires jinja2"
    defines = StringIO()
    for name, val in kwargs.items():
        defines.write(f"    {name} : tl.constexpr = {val}\n")
    defines = defines.getvalue()

    fake_out = ir.Buffer("buf_out", layout) if output_node is None else output_node
    kernel_name = f"triton_{self.name}"

    numel = sympy_product(layout.size)
    buffers = (
        itertools.chain(input_nodes, (fake_out,))
        if output_node is None
        else input_nodes
    )
    if not TritonScheduling.can_use_32bit_indexing(numel, buffers):
        raise NotImplementedError(
            "64-bit indexing is not yet implemented for triton templates"
        )

    kernel_options = dict(
        input_nodes=input_nodes,
        defines=defines,
        num_stages=num_stages,
        num_warps=num_warps,
        grid_fn=self.grid,
        meta=kwargs,
        call_sizes=layout.size,
        prefix_args=prefix_args,
        suffix_args=suffix_args,
        epilogue_fn=epilogue_fn,
        index_dtype="tl.int32",
    )
    with patch.object(
        V.graph, "get_dtype", self._fake_get_dtype(fake_out)
    ), TritonTemplateKernel(
        kernel_name=kernel_name,
        output_node=fake_out,
        use_jit=True,
        **kernel_options,
    ) as kernel:
        try:
            code = kernel.render(self.template, kwargs).finalize()
        except ZeroDivisionError:
            # TODO(nmacchioni): fix sympy division by zero
            return None
        if self.debug:
            print("Generated Code:\n", code)
        extra = (
            "-".join(
                [
                    *[
                        f"{kwarg}={repr(kwargs[kwarg])}"
                        for kwarg in sorted(kwargs.keys())
                    ],
                    f"num_stages={num_stages}",
                    f"num_warps={num_warps}",
                ]
            )
            + "-"
        )
        mod = PyCodeCache.load(code, extra)
        _, call_args, _ = kernel.args.python_argdefs()

    expected_args = list(unique(x.get_name() for x in input_nodes))
    if output_node is None:
        expected_args.extend([fake_out.get_name()])
    assert list(call_args)[: len(expected_args)] == expected_args, (
        call_args,
        expected_args,
    )
    extra_args = V.graph.sizevars.size_hints(
        map(sympy.expand, call_args[len(expected_args) :]),
        fallback=config.unbacked_symint_fallback,
    )

    kernel_hash_name = f"triton_{self.name}_{next(self.index_counter)}"

    def make_kernel_render(out_node):
        kernel = TritonTemplateKernel(
            kernel_name=str(Placeholder.KERNEL_NAME),
            output_node=out_node,
            use_jit=False,
            **kernel_options,
        )
        render = functools.partial(
            kernel.render,
            self.template,
            kwargs,
        )
        return kernel, render

    # create the BenchmarkRequest
    assert mod.__file__ is not None
    grid = self.grid(
        *V.graph.sizevars.size_hints(
            layout.size,
            fallback=config.unbacked_symint_fallback,
        ),
        kwargs,
    )
    bmreq = TritonBenchmarkRequest(
        module_path=mod.__file__,
        module_cache_key=mod.key,
        kernel_name=kernel_name,
        grid=grid,
        extra_args=extra_args,
        num_stages=num_stages,
        num_warps=num_warps,
        matrix_instr_nonkdim=kwargs.get("matrix_instr_nonkdim", 0),
        input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
        output_tensor_meta=TensorMeta.from_irnodes(layout),
    )

    return TritonTemplateCaller(
        kernel_hash_name,
        input_nodes,
        layout,
        make_kernel_render,
        extra.strip("-").replace("-", ", "),
        bmreq,
        log_info={
            "tile_shape": str(
                (
                    kwargs.get("BLOCK_M", -1),
                    kwargs.get("BLOCK_K", -1),
                    kwargs.get("BLOCK_N", -1),
                )
            ),
            "num_stages": num_stages,
            "num_warps": num_warps,
            "allow_tf32": str(kwargs.get("ALLOW_TF32", None)),
            "acc_type": str(kwargs.get("ACC_TYPE", None)),
        },
        mutated_inputs=mutated_inputs,
    )


torch._inductor.select_algorithm.TritonTemplate.generate = generate


def TritonTemplateCaller__init__(
    self,
    name,
    input_nodes,
    layout,
    make_kernel_render,
    debug_extra,
    bmreq,
    log_info: Optional[
        Dict[str, Union[PrimitiveInfoType, List[PrimitiveInfoType]]]
    ] = None,
    mutated_inputs=None,
):
    super(TritonTemplateCaller, self).__init__(name, input_nodes, layout)
    self.make_kernel_render = make_kernel_render
    self.debug_extra = debug_extra
    self.bmreq: TritonBenchmarkRequest = bmreq
    if log_info is None:
        log_info = {}
    self.log_info: Dict[str, Any] = log_info
    self.log_info.update(
        {
            "backend": "Triton",
            "grid": str(self.bmreq.grid),
            "num_stages": self.bmreq.num_stages,
            "num_warps": self.bmreq.num_warps,
        }
    )
    self.mutated_inputs = mutated_inputs


def output_node(self):
    return ir.TensorBox.create(
        ir.TritonTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            debug_extra=self.debug_extra,
            mutated_inputs=self.mutated_inputs,
        )
    )


torch._inductor.select_algorithm.TritonTemplateCaller.__init__ = (
    TritonTemplateCaller__init__
)
torch._inductor.select_algorithm.TritonTemplateCaller.output_node = output_node
