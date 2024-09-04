import torch
from torch._inductor.fx_passes.serialized_patterns.central_index import central_index


def get_serialized_pattern(key):
    import torch._inductor  # noqa: F401
    from torch._inductor import config

    # Modified by Cambricon start: comment below codes.
    # if config.fallback_random:
    #    return None
    # Modified by Cambricon end

    # TODO - could add more validation that the same set of decomps used when
    # tracing SDPA are also used in current context. softmax, dropout, etc
    # decomp use is stable so not an issue in practice.
    return central_index.get(key)


torch._inductor.fx_passes.serialized_patterns.central_index.get_serialized_pattern = (
    get_serialized_pattern
)
