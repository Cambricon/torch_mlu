import functools

import torch
from torch._dynamo.utils import tensortype_to_dtype


@staticmethod
@functools.lru_cache(None)
def _in_graph_classes():
    return set(tensortype_to_dtype.keys()) | {
        torch.Tensor,
        torch.cuda.Stream,
        torch.cuda.Event,
        torch.mlu.Stream,
        torch.mlu.Event,
    }


torch._dynamo.variables.user_defined.UserDefinedObjectVariable._in_graph_classes = (
    _in_graph_classes
)
