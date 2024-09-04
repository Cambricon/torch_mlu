import sys
import torch_mlu
import warnings
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation
from contextlib import contextmanager

def set_flags(_enabled=None, _benchmark=None, _benchmark_limit=None, _deterministic=None, _allow_tf32=None):
    orig_flags = (None, None, None, None,
                  torch_mlu._MLUC._get_cnnl_allow_tf32())
    if _enabled is True:
        warnings.warn("torch.backends.cnnl.enabled is not available on MLU device.")
    if _benchmark is True:
        warnings.warn("torch.backends.cnnl.benchmark is not available on MLU device.")
    if _benchmark_limit != 0 and _benchmark_limit is not None:
        warnings.warn("torch.backends.cnnl.benchmark_limit is not available on MLU device.")
    if _deterministic is True:
        warnings.warn("torch.backends.cnnl.deterministic is not available on MLU device.")
    if _allow_tf32 is not None:
        torch_mlu._MLUC._set_cnnl_allow_tf32(_allow_tf32)
    return orig_flags

@contextmanager
def flags(enabled=False, benchmark=False, benchmark_limit=0, deterministic=False, allow_tf32=True):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, benchmark, benchmark_limit, deterministic, allow_tf32)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)

class CnnlModule(PropModule):
    def __init__(self, m, name):
        super(CnnlModule, self).__init__(m, name)

    # Control wether to allow TF32 on part of CNNL ops,
    # same function as `torch.backends.cudnn.allow_tf32`, currently only affect conv.
    allow_tf32 = ContextProp(torch_mlu._MLUC._get_cnnl_allow_tf32, torch_mlu._MLUC._set_cnnl_allow_tf32)
