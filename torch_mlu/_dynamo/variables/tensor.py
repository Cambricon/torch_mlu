import torch
from torch._dynamo.utils import (
    fqn,
    proxy_args_kwargs,
    tensortype_to_dtype,
)
from torch._dynamo.variables.tensor import TensorVariable
from torch._dynamo.variables.constant import ConstantVariable


def method_attr_is_cuda(self, tx):
    if self.device is not None:
        return ConstantVariable.create(self.device.type == "mlu")


TensorVariable.method_attr_is_cuda = method_attr_is_cuda


def method_type(self, dtype=None, non_blocking=False, **kwargs):
    if (
        dtype is None
        and self.dtype is not None
        and isinstance(self.device, torch.device)
    ):
        tensortype = next(k for k, v in tensortype_to_dtype.items() if self.dtype in v)
        if self.device.type == "mlu":
            return ConstantVariable.create(f"torch.mlu.{tensortype.__name__}")
        else:
            return ConstantVariable.create(f"torch.{tensortype.__name__}")
    elif (
        dtype is not None
        and fqn(type(dtype.as_python_constant())) == "torch.tensortype"
    ):
        # torch.FloatTensor, etc. are all of type "torch.tensortype".
        # torch.fx's tracer fails on these types, because it doesn't support arguments of torch.tensortype type.
        # So, we pass it in as a string (which is also supported, see above implementation for .type() with 0 args)
        tensor_type = dtype.as_python_constant()
        tensor_type_const = ConstantVariable.create(fqn(tensor_type))

        from torch._dynamo.symbolic_convert import InstructionTranslator
        from torch._dynamo.variables.builder import wrap_fx_proxy

        tx = InstructionTranslator.current_tx()

        if non_blocking:
            kwargs = {"non_blocking": non_blocking, **kwargs}

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                "type",
                *proxy_args_kwargs([self, tensor_type_const], kwargs),
            ),
        )


TensorVariable.method_type = method_type
