import logging
import time

from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Union,
)

import torch
from torch._dynamo import (
    config as dynamo_config,
    utils as dynamo_utils,
)
from torch._functorch.aot_autograd import make_boxed_func
from torch._inductor.utils import BoxedBool
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.codecache import CompiledFxGraph, FxGraphCache
from torch._inductor.debug import save_args_for_compile_fx_inner

from torch.fx._lazy_graph_module import _use_lazy_graph_module  # type: ignore[attr-defined]
from torch._inductor.ir import ExternKernelNode
from torch._inductor.debug import DebugContext
from torch._inductor import config
from torch._inductor.utils import (
    get_dtype_size,
    has_incompatible_cudagraph_ops,
    output_node,
)

from torch._inductor.compile_fx import (
    align_inputs,
    align_inputs_from_check_idxs,
    fx_codegen_and_compile,
    complex_memory_overlap,
    copy_misaligned_inputs,
    cudagraphify,
    get_expanded_dims,
    get_patched_config_dict,
    index_expanded_dims_and_copy_,
    remove_unaligned_input_idxs,
    static_input,
    _step_logger,
)

from ..mlu._utils import replace_references

if config.is_fbcode():
    from torch._inductor.fb.utils import time_and_log
else:
    # no-op decorator
    def time_and_log(attr: str, extra_loggings: Optional[Dict[str, str]] = None):
        return dynamo_utils.identity


log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
ALIGNMENT = 16


def get_input_idxs_to_check(
    inputs: Union[List[torch.Tensor], Sequence[int]],
    static_input_idxs: Sequence[int],
) -> Sequence[int]:
    def is_aligned(storage_offset, dtype):
        return (storage_offset * get_dtype_size(dtype)) % ALIGNMENT == 0

    ids_to_check = []
    for i, input in enumerate(inputs):
        if (
            isinstance(input, torch.Tensor)
            and (
                i not in static_input_idxs
                or not is_aligned(input.storage_offset(), input.dtype)
            )
            and input.device.type == "mlu"
        ):
            ids_to_check.append(i)
    return ids_to_check


replace_references(
    torch._inductor.compile_fx.get_input_idxs_to_check, get_input_idxs_to_check
)


def cudagraphify_impl(
    model: torch.fx.GraphModule,
    inputs: List[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
):
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
    static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
    copy_misaligned_inputs(inputs, check_input_idxs)

    assert isinstance(inputs, list)

    inps_expanded_dims = [
        get_expanded_dims(x) if idx not in static_input_idxs else []
        for idx, x in enumerate(inputs)
    ]

    # allocate static tensor inputs
    static_inputs = [
        x
        if not isinstance(x, torch.Tensor)
        else static_input(x)
        if idx not in static_input_idxs
        else x.detach()
        for idx, x in enumerate(inputs)
    ]

    # copy over input values for fresh allocations
    for idx, (x, expanded_dims) in enumerate(zip(inputs, inps_expanded_dims)):
        if isinstance(x, torch.Tensor) and idx not in static_input_idxs:
            index_expanded_dims_and_copy_(static_inputs[idx], x, expanded_dims)

    # warmup
    torch.mlu.synchronize()
    stream = torch.mlu.Stream()
    stream.wait_stream(torch.mlu.current_stream())
    # copy static_inputs because it will be cleared in model
    with torch.mlu.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.mlu.current_stream().wait_stream(stream)
    torch.mlu.synchronize()

    # record
    graph = torch.mlu.MLUGraph()
    with torch.mlu.graph(graph, stream=stream, capture_error_mode="thread_local"):
        static_outputs = model(list(static_inputs))
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(new_inputs):
            assert len(static_inputs) == len(new_inputs)
            for idx, (dst, src, expanded_dims) in enumerate(
                zip(static_inputs, new_inputs, inps_expanded_dims)
            ):
                if not isinstance(dst, torch.Tensor):
                    pass
                elif idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    # TODO - could make one single op of multiple slices
                    # and avoid dispatch.
                    # Could also pre-index the `dst` tensors
                    index_expanded_dims_and_copy_(dst, src, expanded_dims)
            new_inputs.clear()
            graph.replay()
            return static_outputs

    else:
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        def run(new_inputs):
            for idx in copy_indices:
                expanded_dims = inps_expanded_dims[idx]
                index_expanded_dims_and_copy_(
                    static_inputs[idx], new_inputs[idx], expanded_dims
                )
            new_inputs.clear()
            graph.replay()
            return static_outputs

    return align_inputs_from_check_idxs(run, check_input_idxs)


torch._inductor.compile_fx.cudagraphify_impl = cudagraphify_impl


@DebugContext.wrap
@torch.utils._python_dispatch._disable_current_modes()
@time_and_log(
    attr="compilation time (in seconds)",
    extra_loggings={"config_dict": str(get_patched_config_dict())},
)
# Need this decorator for compile_fx_inner even if we already have one for
# compile_fx. The reason is the compilation for backward graph may happen after
# compile_fx return and we may want to use the _LazyGraphModule for compiling
# the backward graph as well.
@_use_lazy_graph_module(dynamo_config.use_lazy_graph_module)
@dynamo_utils.dynamo_timed(phase_name="inductor_compile")
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    cudagraphs: Optional[BoxedBool] = None,
    num_fixed: int = 0,
    is_backward: bool = False,
    graph_id: Optional[int] = None,
    cpp_wrapper: bool = False,
    aot_mode: bool = False,
    is_inference: bool = False,
    boxed_forward_device_index: Optional[BoxedDeviceIndex] = None,
    user_visible_outputs: FrozenSet[str] = frozenset(),
    layout_opt: Optional[bool] = None,
    extern_node_serializer: Optional[Callable[[List[ExternKernelNode]], Any]] = None,
) -> Union[CompiledFxGraph, str]:
    """
    Inductor API that compiles a single graph.

    If you change the argument list for this function, make sure you
    also update the call to save_args_for_compile_fx_inner below accordingly.
    """
    if dynamo_utils.count_calls(gm.graph) == 0 and not aot_mode:
        # trigger the real recompilation for _LazyGraphModule before returning
        # the forward method.
        from torch.fx._lazy_graph_module import _LazyGraphModule

        _LazyGraphModule.force_recompile(gm)
        return make_boxed_func(gm.forward)

    assert isinstance(
        next(iter(reversed(gm.graph.nodes))).args[0], (tuple, list)
    ), f"inductor can only compile FX graphs which return a tuple/list, but got {gm.graph}"

    if config.save_args:
        save_args_for_compile_fx_inner(
            gm,
            example_inputs,
            cudagraphs=cudagraphs,
            num_fixed=num_fixed,
            is_backward=is_backward,
            graph_id=graph_id,
            cpp_wrapper=cpp_wrapper,
            aot_mode=aot_mode,
            is_inference=is_inference,
            boxed_forward_device_index=boxed_forward_device_index,
            user_visible_outputs=user_visible_outputs,
            layout_opt=layout_opt,
        )

    if cudagraphs is None:
        cudagraphs = BoxedBool(config.triton.cudagraphs)

    # Inputs to fx_codegen_and_compile
    # Anything that affects codegen should go here, so if the signature
    # of fx_codegen_and_compile changes, the dict should be updated accordingly
    graph_kwargs = {
        "cudagraphs": cudagraphs,
        "num_fixed": num_fixed,
        "is_backward": is_backward,
        "graph_id": graph_id,
        "cpp_wrapper": cpp_wrapper,
        "aot_mode": aot_mode,
        "is_inference": is_inference,
        "user_visible_outputs": user_visible_outputs,
        "layout_opt": layout_opt,
        "extern_node_serializer": extern_node_serializer,
    }

    start = time.time()

    if config.fx_graph_cache and not aot_mode:
        compiled_graph = FxGraphCache.load(
            fx_codegen_and_compile, gm, example_inputs, graph_kwargs
        )
    else:
        compiled_graph = fx_codegen_and_compile(
            gm, example_inputs, **graph_kwargs  # type: ignore[arg-type]
        )

    log.debug("FX codegen and compilation took %.3fs", time.time() - start)

    # check cudagraph disabling reasons from inductor lowering
    if cudagraphs and compiled_graph.disabled_cudagraphs_reason:
        perf_hint_log.warning(
            "skipping cudagraphs due to %s", compiled_graph.disabled_cudagraphs_reason
        )
        BoxedBool.disable(cudagraphs)

    # Return the output strides to the caller via TracingContext
    context = torch._guards.TracingContext.try_get()
    if context is not None and context.output_strides is not None:
        assert len(context.output_strides) == 0
        context.output_strides.extend(compiled_graph.output_strides)

    if aot_mode:
        return compiled_graph

    if cudagraphs:
        # output args are tuple of first argument
        output = output_node(gm)
        assert len(output.args) == 1
        stack_traces = [
            (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
            for arg in output.args[0]
        ]

        complex_memory_overlap_inputs = any(
            complex_memory_overlap(t)
            for t in example_inputs
            if isinstance(t, torch.Tensor)
        )

        from torch._inductor.cudagraph_utils import check_for_mutation

        has_mutation_str = check_for_mutation(gm, compiled_graph, num_fixed)
        has_mutation = has_mutation_str is not None

        if has_mutation:
            compiled_graph.disabled_cudagraphs_reason = has_mutation_str

        cudagraph_tests = [
            (not has_mutation, "mutated inputs"),
            (not has_incompatible_cudagraph_ops(gm), "incompatible ops"),
            (not complex_memory_overlap_inputs, "complex memory overlap"),
            (
                all(
                    isinstance(t, (torch.Tensor, torch.SymInt)) for t in example_inputs
                ),
                "non-Tensor inputs",
            ),
        ]
        cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]

        if not cudagraph_fail_reasons:
            if not config.triton.cudagraph_trees:
                # Force specialize all inputs so that CUDA graphs will work
                for t in example_inputs:
                    if isinstance(t, torch.SymInt):
                        int(t)  # guard

            if (
                boxed_forward_device_index is not None
                and not is_inference
                and not is_backward
            ):
                boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))

            compiled_graph.current_callable = cudagraphify(
                compiled_graph.get_current_callable(),
                example_inputs,
                static_input_idxs=range(num_fixed),
                device_index=next(iter(compiled_graph.device_idxs)),
                stack_traces=stack_traces,
                is_backward=is_backward,
                is_inference=is_inference,
                constants=tuple(compiled_graph.constants.values()),
            )
        else:
            BoxedBool.disable(cudagraphs)

            # See [Backward Generation Handling]
            # if cudagraph'd the forward and set the device, we need to let the cudagraph manager
            # know we are we running the backward even if we will not run it in cudagraphs
            if is_backward and config.triton.cudagraph_trees:
                assert boxed_forward_device_index is not None
                assert boxed_forward_device_index.value is not None
                compiled_graph_callable = compiled_graph.get_current_callable()

                manager = torch._inductor.cudagraph_trees.get_manager(
                    boxed_forward_device_index.value, create_if_none_exists=False
                )
                # should already exist from forward
                assert manager is not None

                def compiled_artifact(new_inputs):
                    manager.set_to_running_backward()
                    return compiled_graph_callable(new_inputs)

                compiled_graph.current_callable = compiled_artifact

            if "mlu" in compiled_graph.device_types:
                # prefer better disable_cudagraphs_reason bc stack trace
                # TODO: migrate all disable reasons to stack trace, refactor
                if compiled_graph.disabled_cudagraphs_reason:
                    perf_hint_log.warning(compiled_graph.disabled_cudagraphs_reason)
                else:
                    perf_hint_log.warning(
                        "skipping cudagraphs due to %s", cudagraph_fail_reasons
                    )

    # cudagraphs does its own aligning of inputs
    if not cudagraphs:
        new_callable = align_inputs(
            compiled_graph.get_current_callable(), example_inputs, range(num_fixed)
        )
        if new_callable is not compiled_graph.get_current_callable():
            compiled_graph.current_callable = new_callable

    _step_logger()(
        logging.INFO,
        "torchinductor done compiling "
        f"{'BACKWARDS' if is_backward else 'FORWARDS'} "
        f"graph {graph_id}",
    )

    # aot autograd needs to know to pass in inputs as a list
    compiled_graph._boxed_call = True
    return compiled_graph


replace_references(torch._inductor.compile_fx.compile_fx_inner, compile_fx_inner)
