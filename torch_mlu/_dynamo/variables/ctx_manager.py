import inspect

import torch
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.ctx_manager import AutocastModeVariable


@staticmethod
def create(func, args, kwargs):
    assert func in [
        torch.amp.autocast_mode.autocast,
        torch.mlu.amp.autocast,
        torch.cpu.amp.autocast,
    ]
    # device_type : str,
    # dtype : Optional[_dtype] = None,
    # enabled : bool = True,
    # cache_enabled : Optional[bool] = None):cache_enabled
    bound_args = inspect.signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()
    target_values = []
    kwargs.clear()

    for key in ["device_type", "dtype", "enabled", "cache_enabled"]:
        if key == "device_type" and func in [
            torch.mlu.amp.autocast,
            torch.cpu.amp.autocast,
        ]:
            arg = "mlu" if func is torch.mlu.amp.autocast else "cpu"
        else:
            arg = bound_args.arguments[key]
        if isinstance(arg, VariableTracker):
            target_values.append(arg.as_python_constant())
        else:
            target_values.append(arg)

    var = AutocastModeVariable(target_values, initial_values=None, **kwargs)
    return var


AutocastModeVariable.create = create
