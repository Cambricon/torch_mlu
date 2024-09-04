import inspect
import logging

from typing import Dict, List

import torch
from torch._logging import warning_once
from torch._streambase import _StreamBase
from torch._dynamo import variables
from torch._dynamo.utils import guard_if_dyn
from torch._dynamo.variables.torch import (
    supported_ctx_manager_classes,
    constant_fold_functions,
)
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.ctx_manager import (
    AutocastModeVariable,
    NullContextVariable,
    TorchFunctionDisableVariable,
)

log = logging.getLogger(__name__)

supported_ctx_manager_classes.add(torch.mlu.amp.autocast)
constant_fold_functions.append(torch.mlu.get_device_properties)
constant_fold_functions.append(torch.mlu.is_available)


def call_function(
    self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
) -> "VariableTracker":
    from torch._dynamo.variables import (
        DisabledSavedTensorsHooksVariable,
        GradIncrementNestingCtxManagerVariable,
        GradInplaceRequiresGradCtxManagerVariable,
        GradModeVariable,
        InferenceModeVariable,
        StreamVariable,
        VmapIncrementNestingCtxManagerVariable,
    )

    if self.value is torch.no_grad:
        if len(args) == 1 and isinstance(
            args[0], variables.functions.BaseUserFunctionVariable
        ):
            ctx = GradModeVariable.create(tx, False)
            return ctx.call_function(tx, args, kwargs)
        else:
            return GradModeVariable.create(tx, False)
    elif self.value is torch.enable_grad:
        if len(args) == 1 and isinstance(
            args[0], variables.functions.BaseUserFunctionVariable
        ):
            ctx = GradModeVariable.create(tx, True)
            return ctx.call_function(tx, args, kwargs)
        return GradModeVariable.create(tx, True)
    elif self.value is torch.set_grad_enabled and len(args) == 1:
        return GradModeVariable.create(
            tx, args[0].as_python_constant(), initialized=True
        )
    elif self.value is torch.inference_mode:
        assert len(args) <= 1 and len(kwargs) == 0
        inf_mode = args[0].as_python_constant() if len(args) == 1 else True
        return InferenceModeVariable.create(tx, inf_mode)
    elif inspect.isclass(self.value) and issubclass(self.value, _StreamBase):
        from torch._dynamo.variables.builder import wrap_fx_proxy_cls

        return wrap_fx_proxy_cls(
            StreamVariable,
            tx,
            tx.output.create_proxy(
                "call_function",
                self.value,
                (),
                {},
            ),
        )
    elif self.value in (
        torch.amp.autocast_mode.autocast,
        torch.mlu.amp.autocast,
        torch.cpu.amp.autocast,
    ):
        return AutocastModeVariable.create(self.value, args, kwargs)
    elif self.value in (
        torch.profiler.profile,
        torch.profiler.record_function,
        torch.autograd.profiler.profile,
        torch.autograd.profiler.record_function,
    ):
        warning_once(log, "Profiler function %s will be ignored", self.value)
        return NullContextVariable()
    elif self.value is torch._C.DisableTorchFunctionSubclass:
        assert not (args or kwargs)
        return TorchFunctionDisableVariable.create(tx)
    elif self.value is torch._functorch.vmap.vmap_increment_nesting:
        assert len(args) == 2
        return VmapIncrementNestingCtxManagerVariable.create(
            tx,
            [guard_if_dyn(x) for x in args],
        )
    elif self.value is torch._functorch.eager_transforms.grad_increment_nesting:
        assert len(args) == 0
        return GradIncrementNestingCtxManagerVariable.create(tx)
    elif self.value is torch._functorch.eager_transforms.enable_inplace_requires_grad:
        assert len(args) == 1
        return GradInplaceRequiresGradCtxManagerVariable.create(
            tx,
            [guard_if_dyn(x) for x in args],
        )
    elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
        assert len(args) == 1
        return DisabledSavedTensorsHooksVariable.create(
            tx, args[0].as_python_constant()
        )


torch._dynamo.variables.torch.TorchCtxManagerClassVariable.call_function = call_function
