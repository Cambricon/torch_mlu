# All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
# All rights reserved.
# All other contributions:
# Copyright (c) 2014--2022, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List, Optional, Tuple, Union
import warnings

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import torch

# A workaround to support both TorchScript and MyPy:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    # The JIT doesn't understand Union, nor torch.dtype here
    DType = int

from torch._jit_internal import boolean_dispatch, BroadcastingList2, BroadcastingList3
from torch.overrides import has_torch_function_unary, handle_torch_function
from torch.nn.modules.utils import _pair, _list_with_default

import torch.nn.functional as F
from torch.nn.functional import _unpool_output_size

Tensor = torch.Tensor


def max_unpool2d(
    input: Tensor,
    indices: Tensor,
    kernel_size: BroadcastingList2[int],
    stride: Optional[BroadcastingList2[int]] = None,
    padding: BroadcastingList2[int] = 0,
    output_size: Optional[BroadcastingList2[int]] = None,
) -> Tensor:
    r"""Compute a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(
            max_unpool2d,
            (input,),
            input,
            indices,
            kernel_size,
            stride=stride,
            padding=padding,
            output_size=output_size,
        )
    kernel_size = _pair(kernel_size)
    if stride is not None:
        _stride = _pair(stride)
    else:
        _stride = kernel_size
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    # TODO(PYTORCH-10053): revert this change once cnnl supported same implementation as cuda
    if input.device.type == "mlu":
        return torch.ops.torch_mlu.max_unpool2d(
            input, indices, kernel_size, _stride, padding, output_size
        )
    else:
        return torch._C._nn.max_unpool2d(input, indices, output_size)


def apply_functional_patch():
    F.max_unpool2d.__code__ = max_unpool2d.__code__
