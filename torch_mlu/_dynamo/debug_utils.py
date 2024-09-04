# mypy: disable-error-code="method-assign"

import functools
import subprocess
import textwrap
from collections import Counter

import torch
import torch_mlu

from torch._dynamo import debug_utils

MAX_CONSTANT_NUMEL_INLINE = 4


@staticmethod
def convert(gm):
    # Modified by Cambricon: replace the hardcode 'cuda' with 'mlu'
    from torch.nn.modules.module import _addindent

    tab = " " * 4

    model_str = textwrap.dedent(
        """
        from torch.nn import *
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
        """
    )

    for module_name, module in gm.named_children():
        module_str = f"{module.__repr__()}"
        # module should be a core torch.nn.Module, so all parameters
        # should be on the same device.
        example_param = next(module.parameters(), None)
        if example_param is not None and example_param.is_mlu:
            module_str = f"{module_str}.mlu()"
        model_str += f"{tab*2}self.{module_name} = {module_str}\n"

    for buffer_name, buffer in gm._buffers.items():
        if buffer is None:
            continue
        # Serialize full data for small buffers
        if buffer.numel() <= MAX_CONSTANT_NUMEL_INLINE:
            from torch._tensor_str import PRINT_OPTS

            assert PRINT_OPTS.threshold >= MAX_CONSTANT_NUMEL_INLINE
            tensor_str = repr(buffer)
        elif torch.is_floating_point(buffer):
            tensor_str = f"torch.randn({list(buffer.shape)}, dtype={buffer.dtype})"
        else:
            tensor_str = (
                f"torch.randint(1, size={list(buffer.shape)}, dtype={buffer.dtype})"
            )
        if buffer.is_mlu:
            tensor_str = f"{tensor_str}.mlu()"
        model_str += f"{tab*2}self.register_buffer('{buffer_name}', {tensor_str})\n"

    for param_name, param in gm._parameters.items():
        if param is None:
            continue
        maybe_device = ""
        if param.is_mlu:
            maybe_device = ', device="mlu"'
        tensor_str = f"torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}{maybe_device}))"
        model_str += f"{tab*2}self.{param_name} = {tensor_str}\n"

    # TODO - Keep this code for now. But, I don't think we will need this.
    # attrs = dir(gm)
    # for attr in attrs:
    #     if "_tensor_constant" in attr:
    #         val = getattr(gm, attr)
    #         model_str += f"    {attr} = {val!r}\n"

    model_str += f"{_addindent(gm.code, 4)}\n"
    return model_str


torch._dynamo.debug_utils.NNModuleToString.convert = convert


@functools.lru_cache(None)  # subprocess is expensive
def _mlu_system_info_comment():
    if not torch.mlu.is_available():
        return "# torch.mlu.is_available()==False, no MLU info collected\n"

    model_str = "# MLU Info: \n"
    try:
        mlu_version_out = subprocess.check_output(["cncc", "--version"])
        mlu_version_lines = mlu_version_out.decode().split("\n")
        comment = "".join([f"# {s} \n" for s in mlu_version_lines if s not in [""]])
        model_str += f"{comment}\n"
    except (FileNotFoundError, subprocess.CalledProcessError):
        model_str += "# cncc not found\n"

    mlu_names = Counter(
        torch.mlu.get_device_name(i) for i in range(torch.mlu.device_count())
    )

    model_str += "# MLU Hardware Info: \n"
    for name, count in mlu_names.items():
        model_str += f"# {name} : {count} \n"
    model_str += "\n"
    return model_str


def generate_config_string(*, stable_output=False):
    import torch._functorch.config
    import torch._inductor.config

    print("generate_config_string")
    if stable_output:
        return "# config omitted due to stable_output=True"

    experimental_config = torch.fx.experimental._config.codegen_config()  # type: ignore[attr-defined]
    # Modified by Cambricon: add two lines containing 'mlu' word
    return f"""\
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

#from torch_mlu._inductor.fx_passes.mlu_post_pass import mlu_post_pass
#torch._inductor.config.post_grad_custom_post_pass = mlu_post_pass

{torch._dynamo.config.codegen_config()}
{torch._inductor.config.codegen_config()}
{torch._functorch.config.codegen_config()}
{experimental_config}
"""


torch._dynamo.debug_utils.generate_config_string = generate_config_string
