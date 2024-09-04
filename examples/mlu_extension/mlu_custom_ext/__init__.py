import os
import warnings
import torch
import torch_mlu
from .extension import _HAS_OPS
from mlu_custom_ext import ops
if not _HAS_OPS and os.path.dirname(os.path.realpath(__file__)) == os.path.join(
    os.path.realpath(os.getcwd()), "mlu_custom_ext"
):
    message = (
        "You are importing mlu_custom_ext within its own root folder ({}). "
        "This is not expected to work and may give errors. Please exit the "
        "mlu_custom_ext project source and relaunch your python interpreter."
    )
    warnings.warn(message.format(os.getcwd()))
