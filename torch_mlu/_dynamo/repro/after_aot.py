import copy
import functools
import logging
import textwrap
from typing import Any

import torch
import torch_mlu
import torch.fx as fx
from torch._dynamo import config
from torch._dynamo.debug_utils import (
    AccuracyError,
    extra_imports,
    generate_config_string,
    InputWriter,
    NNModuleToString,
    same_two_models,
)
from torch._dynamo.repro.after_aot import (
    backend_aot_accuracy_fails,
    dump_compiler_graph_state,
    dump_to_minify,
    isolate_fails,
    repro_common,
    ACCURACY_FAILS,
)
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets

from ..debug_utils import _mlu_system_info_comment
from torch_mlu.mlu._utils import replace_references

log = logging.getLogger(__name__)


def wrap_compiler_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstraction, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        from torch._subclasses import FakeTensorMode

        compiler_fn = functools.partial(unconfigured_compiler_fn, **kwargs)

        from torch._functorch.aot_autograd import get_aot_graph_name

        graph_name = get_aot_graph_name()

        # TODO: why do we need to deepcopy the original graph?
        orig_graph = copy.deepcopy(gm.graph)
        assert config.repro_after in ("dynamo", "aot", None)

        try:
            # Call the compiler_fn - which is either aot_autograd or inductor
            # with fake inputs
            inner_compiled_fn = compiler_fn(gm, example_inputs)
        except Exception as e:
            # TODO: Failures here are troublesome because no real inputs,
            # need a different serialization strategy
            if config.repro_after == "aot":
                if config.repro_level == 1:
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                elif config.repro_level == 2:
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                log.error("CompilerError")
            raise

        # We may run regular PyTorch compute that may trigger Dynamo, do NOT
        # recursively attempt to accuracy minify in that case!
        def deferred_for_real_inputs(real_inputs):
            # This is a bit obscure: if we recursively try to accuracy minify
            # the SAME function, this would trigger.  But most of the time
            # we should never hit this branch
            if config.repro_after != "aot":
                return inner_compiled_fn(real_inputs)
            with config.patch(repro_after=None):
                return inner_debug_fn(real_inputs)

        def inner_debug_fn(real_inputs):
            """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actually compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deferred.
            """
            # Copy the tensor attrs like shape, stride etc by converting to Fake Tensor
            # because inductor clears the tensor list in its codegen. And example_inputs
            # are available only for the first invocation.
            fake_mode = FakeTensorMode()
            copy_tensor_attrs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in real_inputs
            ]
            if config.repro_level == 3:
                # Always dump the original module in case we have segfaults
                dump_to_minify(
                    fx.GraphModule(gm, orig_graph), real_inputs, compiler_name
                )

            if config.repro_level == 4:
                if compiler_name != "inductor":
                    raise NotImplementedError(
                        "Accuracy minification is supported for inductor only"
                    )
                if backend_aot_accuracy_fails(gm, real_inputs, compiler_fn):
                    log.warning(
                        "Accuracy failed for the AOT Autograd graph %s", graph_name
                    )
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    raise AccuracyError("Bad accuracy detected")
                else:
                    # Call the compiled function with real inputs
                    return inner_compiled_fn(real_inputs)
            else:
                try:
                    # Call the compiled function with real inputs
                    out = inner_compiled_fn(real_inputs)
                    # sync cuda kernels to ensure IMA detection
                    for arg in example_inputs:
                        if isinstance(arg, torch.Tensor) and arg.is_cuda:
                            # Modified by Cambricon start: replace with new codes
                            torch.mlu.synchronize()
                            # Original codes:
                            # torch.cuda.synchronize()
                            # Modified by Cambricon end
                            break
                    return out
                except Exception as e:
                    if config.repro_level == 1:
                        dump_compiler_graph_state(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    elif config.repro_level == 2:
                        dump_to_minify(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    raise

        if config.repro_after == "aot":
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True  # type: ignore[attr-defined]
            return compiled_fn
        else:
            return inner_compiled_fn

    return debug_wrapper


replace_references(
    torch._dynamo.repro.after_aot.wrap_compiler_debug, wrap_compiler_debug
)


def generate_compiler_repro_string(gm, args, *, stable_output=False, save_dir=None):
    # Modified by Cambricon: add a new line: import torch_mlu
    model_str = textwrap.dedent(
        f"""
import torch
import torch_mlu
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

{generate_config_string(stable_output=stable_output)}

isolate_fails_code_str = None

{extra_imports}

        """
    )
    if not stable_output:
        model_str += f"# torch version: {torch.version.__version__}\n"
        # Modified by Cambricon: replace with new codes
        if hasattr(torch.version, "mlu"):
            model_str += f"# torch mlu version: {torch.version.mlu}\n"
        # Original codes:
        # if hasattr(torch.version, "cuda"):
        #     model_str += f"# torch cuda version: {torch.version.cuda}\n"
        # Modified by Cambricon end
        if hasattr(torch.version, "git_version"):
            model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
        # Modified by Cambricon: replace with new codes
        model_str += _mlu_system_info_comment()
        # Original codes:
        # model_str += _cuda_system_info_comment()
        # Modified by Cambricon end

    model_str += NNModuleToString.convert(gm)

    # get hint shape/stride when dynamic shape enabled
    def hint_if_symint(x):
        return tuple(i.node.hint if isinstance(i, torch.SymInt) else i for i in x)

    writer = InputWriter(save_dir)
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            writer.tensor(placeholder, arg)
        else:
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")

    model_str += "\n".join(writer.lines()) + "\n"

    model_str += "mod = Repro()\n"
    return model_str


torch._dynamo.repro.after_aot.generate_compiler_repro_string = (
    generate_compiler_repro_string
)


def inductor_fails(fx_g, args, check_str=None):
    # Modified by Cambricon, replace the 'cuda' with 'mlu'
    has_mlu = False
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_mlu:
            has_mlu = True
            break

    def sync():
        if has_mlu:
            # Ensures that segfaults are surfaced
            torch.mlu.synchronize()

    # Modified by Cambricon end

    from torch._inductor.compile_fx import compile_fx_inner

    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any(isinstance(x, (tuple, list)) for x in result)
    except Exception:
        return False

    sync()

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod(args)
        sync()
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


replace_references(torch._dynamo.repro.after_aot.inductor_fails, inductor_fails)


def repro_minify(options, mod, load_args):
    from functorch.compile import minifier

    mod, args = repro_common(options, mod, load_args)
    compiler_name = "inductor_accuracy" if options.accuracy != "" else "inductor"

    # Modified by Cambricon: replace with new codes
    favored_device = 1 if torch.mlu.device_count() >= 2 else 0
    env_variables = {"MLU_VISIBLE_DEVICES": str(favored_device)}
    # Original codes:
    # favored_device = 1 if torch.cuda.device_count() >= 2 else 0
    # env_variables = {"CUDA_VISIBLE_DEVICES": str(favored_device)}
    # Modified by Cambricon end

    module_fails: Any
    if options.isolate:
        module_fails = functools.partial(
            isolate_fails,
            env=env_variables,
            compiler_name=compiler_name,
            save_dir=options.save_dir,
            accuracy=options.accuracy,
            tracing_mode=options.tracing_mode,
        )
    else:
        module_fails = ACCURACY_FAILS[options.accuracy]

    minifier(
        mod,
        args,
        module_fails=functools.partial(module_fails, check_str=options.check_str),
        dump_state=functools.partial(
            dump_compiler_graph_state, compiler_name=compiler_name
        ),
        save_dir=options.save_dir,
        offload_to_disk=options.offload_to_disk,
        skip_offload=options.skip_saving_eager_intermediates,
        skip_sanity=options.skip_sanity,
        max_granularity=options.max_granularity,
    )


torch._dynamo.repro.after_aot.repro_minify = repro_minify


def repro_run(options, mod, load_args):
    from torch._inductor.compile_fx import compile_fx_inner

    mod, args = repro_common(options, mod, load_args)

    # Modified by Cambricon: replace with new codes
    from torch.mlu import synchronize

    # Original codes:
    # from torch.cuda import synchronize
    # Modified by Cambricon end

    compiled = compile_fx_inner(mod, args)

    if options.accuracy != "":
        # We don't really respect --accuracy vs --strict-accuracy here, it
        # seems counterintuitive
        if not same_two_models(mod, compiled, args, only_fwd=True):
            raise AccuracyError("Bad accuracy detected")
    else:
        need_sync = False
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                need_sync = True
                break
        ref = compiled(list(args))
        if need_sync:
            synchronize()  # ensure segfaults are surfaced
    return lambda: compiled(list(args))


torch._dynamo.repro.after_aot.repro_run = repro_run
