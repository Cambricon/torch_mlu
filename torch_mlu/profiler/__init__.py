import torch
from ._pattern_matcher import (
    ExtraMLUCopyPattern,
    FP32MatMulPattern,
    MatMulDimInFP16Pattern,
    report_all_anti_patterns
)

from .profiler import emit_cnpx
torch.autograd.profiler.__setattr__("emit_cnpx", emit_cnpx)

# Add torch.profiler.ProfilerActivity.MLU and mapping it to torch.profiler.ProfilerActivity.PrivateUse1
torch.profiler.ProfilerActivity.MLU = torch.profiler.ProfilerActivity.PrivateUse1

torch.profiler._pattern_matcher.__setattr__("ExtraMLUCopyPattern", ExtraMLUCopyPattern)
torch.profiler._pattern_matcher.__setattr__("FP32MatMulPattern", FP32MatMulPattern)
torch.profiler._pattern_matcher.__setattr__("MatMulDimInFP16Pattern", MatMulDimInFP16Pattern)
torch.profiler._pattern_matcher.__setattr__("report_all_anti_patterns", report_all_anti_patterns)

# TODO(fuwenguang): Remove the following code once landing these into the native community.
# Support passing 'self_mlu_xxx's and mapping them to 'self_privateuse1_xxx's
from functools import wraps
def replace_mlu_to_privateuse1(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if kwargs:
            sort_by = kwargs.get('sort_by', None)
            if sort_by:
                kwargs['sort_by'] = sort_by.replace("mlu", "privateuse1")

            metric = kwargs.get('metric', None)
            if metric:
                kwargs['metric'] = metric.replace("mlu", "privateuse1")
        return fn(*args, **kwargs)

    return wrapper_fn

_build_table_fn = getattr(torch.autograd.profiler_util, "_build_table")
_new_build_table = replace_mlu_to_privateuse1(_build_table_fn)
setattr(torch.autograd.profiler_util, "_build_table", _new_build_table)

_export_stacks = getattr(torch.autograd.profiler_util.EventList, "export_stacks")
_new_export_stacks = replace_mlu_to_privateuse1(_export_stacks)
setattr(torch.autograd.profiler_util.EventList, "export_stacks", _new_export_stacks)
