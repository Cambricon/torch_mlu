import torch
from torch.distributed.algorithms._comm_hooks.default_hooks import LowPrecisionState


def _decompress(state: LowPrecisionState, grad: torch.Tensor):
    """
    Casts gradients back to full parameter precision so that further computation happens in full precision.
    """
    orig_grad_data = grad.data
    grad.data = grad.data.to(state.parameter_type)
    # Don't let this memory get reused until after the transfer.
    orig_grad_data.record_stream(torch.mlu.current_stream())  # type: ignore[arg-type]


def apply_default_hook_patch():
    torch.distributed.algorithms._comm_hooks.default_hooks._decompress = _decompress
