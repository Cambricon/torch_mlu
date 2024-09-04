from torch._dynamo.trace_rules import (
    BUILTIN_SKIPLIST,
    SKIP_DIRS,
    check_file,
    _module_dir,
    _recompile_re,
    lookup_inner,
    is_aten_op_or_tensor_method,
    get_torch_obj_rule_map,
)
from torch._dynamo.variables import (
    TorchInGraphFunctionVariable,
    SkipFunctionVariable,
    UserFunctionVariable,
)

from torch._dynamo.utils import getfile, hashable
import torch_mlu
from torch_mlu.mlu._utils import replace_references

MLU_BUILTIN_SKIPLIST = (torch_mlu,)
BUILTIN_SKIPLIST = BUILTIN_SKIPLIST + MLU_BUILTIN_SKIPLIST
SKIP_DIRS.extend(filter(None, (_module_dir(m) for m in BUILTIN_SKIPLIST)))
_recompile_re()


def mlu_lookup_inner(obj, name=None, filename=None, is_direct_call=True):
    # Step 1: lookup obj's tracing rule in `torch_name_rule_map`.
    # The rules defined in `torch_name_rule_map` mainly includes two parts:
    # - Manually defined rules for any functions.
    # - The list of torch in graph functions.
    if not hashable(obj):
        return None
    if obj is not None:
        if is_aten_op_or_tensor_method(obj):
            return TorchInGraphFunctionVariable
        rule = get_torch_obj_rule_map().get(obj, None)
        # Enabling model_transfer changes the function type from <built-in method **>
        # to <_VariableFunctionsClass.**> wrapping <built-in method **>
        if (
            rule is None
            and hasattr(obj, "is_mlu_model_transfer")
            and obj.is_mlu_model_transfer is True
            and hasattr(obj, "__wrapped__")
        ):
            rule = get_torch_obj_rule_map().get(obj.__wrapped__, None)
        if rule is not None:
            return rule

    # Step 2: lookup obj's tracing rule by function name.
    if is_direct_call:
        if name == "patched_init":
            return SkipFunctionVariable
        elif name == "__torch_function__":
            return UserFunctionVariable

    # Step 3: lookup obj's tracing rule by filename.
    if filename is None:
        filename = getfile(obj)

    if check_file(filename, is_direct_call).skipped:
        return SkipFunctionVariable
    else:
        return UserFunctionVariable


replace_references(lookup_inner, mlu_lookup_inner)
