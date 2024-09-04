import sys
import torch
import torch_mlu
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def set_flags(_enabled):
    orig_flags = (torch_mlu._MLUC._get_mlufusion_enabled(),)
    torch_mlu._MLUC._set_mlufusion_enabled(_enabled)
    return orig_flags

@contextmanager
def flags(enabled=False):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])

class MlufusionModule(PropModule):
    def __init__(self, m, name):
        super(MlufusionModule, self).__init__(m, name)

    enabled = ContextProp(torch_mlu._MLUC._get_mlufusion_enabled,
      torch_mlu._MLUC._set_mlufusion_enabled)
