from __future__ import print_function

import torch
import torch_mlu


def _enable_mlu_profiler():
    torch_mlu._MLUC._enable_mlu_profiler()
