import torch    # pylint: disable=W0611
import warnings
import torch_mlu
import contextlib
from typing_extensions import deprecated

__all__ = ["matmul", "custom", "CnnlMatmulTF32Controller", "MLUCustomTF32Controller",
  "fake_set_cublas_allow_tf32", "fake_set_cublas_allow_fp16_reduced_precision_reduction"]

class cnFFTPlanCacheAttrContextProp:
    # Like regular ContextProp, but uses the `.device_index` attribute from the
    # calling object as the first argument to the getter and setter.
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter(obj.device_index)

    def __set__(self, obj, val):
        if isinstance(self.setter, str):
            raise RuntimeError(self.setter)
        self.setter(obj.device_index, val)

class cnFFTPlanCache:
    r"""
    Represent a specific plan cache for a specific `device_index`.

    The attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cnFFT plan cache.
    """

    def __init__(self, device_index):
        self.device_index = device_index

    size = cnFFTPlanCacheAttrContextProp(
        torch.ops.torch_mlu._cnfft_get_plan_cache_size,
        ".size is a read-only property showing the number of plans currently in the "
        "cache. To change the cache capacity, set cnfft_plan_cache.max_size.",
    )

    max_size = cnFFTPlanCacheAttrContextProp(
        torch.ops.torch_mlu._cnfft_get_plan_cache_max_size, torch.ops.torch_mlu._cnfft_set_plan_cache_max_size
    )

    def clear(self):
        return torch.ops.torch_mlu._cnfft_clear_plan_cache(self.device_index)

class cnFFTPlanCache(cnFFTPlanCache):
    r"""
    Represents a specific plan cache for a specific `device_index`. The
    attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cnFFT plan cache.
    """
    size = cnFFTPlanCacheAttrContextProp(
        torch.ops.torch_mlu._cnfft_get_plan_cache_size,
         '.size is a read-only property showing the number of plans currently in the '
         'cache. To change the cache capacity, set cnfft_plan_cache.max_size.')


class cnFFTPlanCacheManager(object):
    r"""
    Represents all cnFFT plan caches. When indexed with a device object/index,
    this object returns the `cnFFTPlanCache` corresponding to that device.

    Finally, this object, when used directly as a `cnFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cnFFT plan cache is
    used.
    """

    __initialized = False

    def __init__(self):
        self.caches = []
        self.__initialized = True

    def __getitem__(self, device):
        index = torch.mlu._utils._get_device_index(device)
        dev_cnt = torch.mlu.device_count() if hasattr(torch.mlu, "device_count") else 0
        if index < 0 or index >= dev_cnt:
            raise RuntimeError(
                ("cnfft_plan_cache: expected 0 <= device index < {}, but got "
                 "device with index {}").format(dev_cnt, index))
        if len(self.caches) == 0:
            self.caches.extend(cnFFTPlanCache(index) for index in range(dev_cnt))
        return self.caches[index]

    def __getattr__(self, name):
        return getattr(self[torch.mlu.current_device()], name)

    def __setattr__(self, name, value):
        if self.__initialized:
            return setattr(self[torch.mlu.current_device()], name, value)
        else:
            return super(cnFFTPlanCacheManager, self).__setattr__(name, value)

class CnnlMatmulTF32Controller:
    r"""
    Control wether to allow TF32 on matmul, same function as `torch.backends.cuda.matmul.allow_tf32`.
    """
    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch_mlu._MLUC._get_cnmatmul_allow_tf32()
        raise AssertionError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return torch_mlu._MLUC._set_cnmatmul_allow_tf32(value)
        raise AssertionError("Unknown attribute " + name)

class MLUCustomTF32Controller:
    r"""
    Control wether to allow TF32 on the rest MLU ops, not controlled by
    `CnnlMatmulTF32Controller` and `CnnlTF32Controller`.
    """
    def __getattr__(self, name):
        assert name == "allow_tf32", "Unknown attribute " + name
        return torch_mlu._MLUC._get_mlu_custom_allow_tf32()

    def __setattr__(self, name, value):
        assert name == "allow_tf32", "Unknown attribute " + name
        if not isinstance(value, bool):
            raise  RuntimeError("set_mlu_custom_allow_tf32 expects a bool, "
                "but got {}".format(type(value)))
        return torch_mlu._MLUC._set_mlu_custom_allow_tf32(value)

# Because the float32_matmul_precision flag is device independent,
# we need to prevent torch._C._set_cublas_allow_tf32 from modifying the flg.
def fake_set_cublas_allow_tf32(value):
    if not isinstance(value, bool):
        raise  RuntimeError("set_allow_tf32_cublas expects a bool, "
            "but got {}".format(type(value)))
    warnings.warn("When using MLU device, the cuda API does not take effect. "
      "Please use torch.backends.mlu.matmul.allow_tf32.")

# MLU side is not support matmul using compute type float16, so we
# need to prevent torch._C.allow_fp16_reduced_precision_reduction to modifying the flg.
def fake_set_cublas_allow_fp16_reduced_precision_reduction(value):
    if not isinstance(value, bool):
        raise  RuntimeError("set_allow_tf32_cublas expects a bool, "
            "but got {}".format(type(value)))
    warnings.warn("When using MLU device, the cublas_allow_fp16_reduced_precision_reduction"
      " API does not take effect. And MLU only support compute type float32.")

from torch_mlu._MLUC import _SDPAParams as SDPAParams
from torch.nn.attention import SDPBackend

# Set the __module__ attribute
SDPAParams.__module__ = "torch.backends.mlu"
SDPAParams.__name__ = "SDPAParams"

def flash_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether flash scaled dot product attention is enabled or not.
    """
    return torch._C._get_flash_sdp_enabled()

def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """
    torch._C._set_sdp_use_flash(enabled)

def mem_efficient_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
    return torch._C._get_mem_efficient_sdp_enabled()

def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
    torch._C._set_sdp_use_mem_efficient(enabled)

def math_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether math scaled dot product attention is enabled or not.
    """
    return torch._C._get_math_sdp_enabled()

def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
    torch._C._set_sdp_use_math(enabled)

def can_use_flash_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if FlashAttention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why FlashAttention could not be run.
            Defaults to False.

    Returns:
        True if FlashAttention can be used with the given parameters; otherwise, False.
    """
    return torch_mlu._MLUC._can_use_flash_attention(params, debug)

def can_use_efficient_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if efficient_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why efficient_attention could not be run.
            Defaults to False.

    Returns:
        True if efficient_attention can be used with the given parameters; otherwise, False.
    """
    return torch_mlu._MLUC._can_use_mem_efficient_attention(params, debug)

@contextlib.contextmanager
@deprecated(
    (
        "`torch.backends.mlu.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    category=FutureWarning,
)
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = False
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    from torch.nn.attention import sdpa_kernel

    backend_list = []
    if enable_flash:
        backend_list.append(SDPBackend.FLASH_ATTENTION)
    if enable_mem_efficient:
        backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    if enable_math:
        backend_list.append(SDPBackend.MATH)
    if enable_cudnn:
        warnings.warn((
            "torch_mlu.backends.mlu.sdp_kernel() dont support enable_cudnn=True."))

    with sdpa_kernel(backend_list) as context:
        try:
            yield context
        finally:
            pass

cnfft_plan_cache = cnFFTPlanCacheManager()
matmul = CnnlMatmulTF32Controller()
custom = MLUCustomTF32Controller()
