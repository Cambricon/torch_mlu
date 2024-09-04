import functools
from typing import Dict, Callable
import torch
from torch.overrides import _disable_user_warnings
from torch.overrides import get_testing_overrides as native_get_testing_overrides


@functools.lru_cache(None)
@_disable_user_warnings
def get_testing_overrides() -> Dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    Returns
    -------
    Dict[Callable, Callable]
        A dictionary that maps overridable functions in the PyTorch API to
        lambda functions that have the same signature as the real function
        and unconditionally return -1. These lambda functions are useful
        for testing API coverage for a type that defines ``__torch_function__``.

    Examples
    --------
    >>> import inspect
    >>> my_add = torch.overrides.get_testing_overrides()[torch.add]
    >>> inspect.signature(my_add)
    <Signature (input, other, out=None)>
    """

    ret = native_get_testing_overrides()

    Tensor = torch.Tensor
    ret[Tensor.mlu] = lambda self, memory_format=torch.preserve_format: -1
    ret[Tensor.is_mlu.__get__] = lambda self: -1  # noqa: B009

    return ret


def apply_torch_overrides_patch():
    torch.overrides.get_testing_overrides = get_testing_overrides
