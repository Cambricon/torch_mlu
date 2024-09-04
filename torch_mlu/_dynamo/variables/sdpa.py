import torch
from torch._dynamo.variables.sdpa import SDPAParamsVariable


@staticmethod
def create(tx, value, source):
    from torch_mlu.backends.mlu import SDPAParams
    from torch._dynamo.source import AttrSource
    from torch._dynamo.variables.builder import VariableBuilder
    from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

    query_var = VariableBuilder(tx, AttrSource(source, "query"))(value.query)
    key_var = VariableBuilder(tx, AttrSource(source, "key"))(value.key)
    value_var = VariableBuilder(tx, AttrSource(source, "value"))(value.value)
    attn_mask_var = VariableBuilder(tx, AttrSource(source, "attn_mask"))(
        value.attn_mask
    )
    dropout_var = VariableBuilder(tx, AttrSource(source, "dropout"))(value.dropout)
    is_causal_var = VariableBuilder(tx, AttrSource(source, "is_causal"))(
        value.is_causal
    )
    param_vars = [
        query_var,
        key_var,
        value_var,
        attn_mask_var,
        dropout_var,
        is_causal_var,
    ]
    return TorchInGraphFunctionVariable(SDPAParams).call_function(tx, param_vars, {})


SDPAParamsVariable.create = create


@staticmethod
def is_sdpa_params(value):
    from torch_mlu.backends.mlu import SDPAParams


SDPAParamsVariable.is_sdpa_params = is_sdpa_params
