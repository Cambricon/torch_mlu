import torch

try:
    from ._transform_bias_rescale_qkv_impl import _transform_bias_rescale_qkv_mlu

    aten_lib = torch.library.Library("aten", "IMPL")
    aten_lib.impl("_transform_bias_rescale_qkv", _transform_bias_rescale_qkv_mlu, "PrivateUse1")
except ModuleNotFoundError:
    pass
