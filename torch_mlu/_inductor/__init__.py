import torch

from . import ir
from . import config
from . import utils

from .codegen import common
from .fx_passes.serialized_patterns import central_index

from . import wrapper_benchmark
from . import dependencies
from . import codecache
from . import lowering
from . import triton_heuristics
from . import autotune_process
from . import select_algorithm
from . import decomposition
from . import compile_fx
from . import cudagraph_trees
from . import cudagraph_utils
from .fx_passes import post_grad

# when the performance of triton matmul is good enough, open below lines.
# from .kernel import mm_common
# from .kernel import mm_plus_mm
# from .kernel import bmm
from .kernel import group_norm

from . import graph

if torch._inductor.config.compile_threads > 1:
    from torch._inductor.codecache import AsyncCompile, shutdown_compile_workers
    shutdown_compile_workers()
    AsyncCompile.process_pool.cache_clear()
    AsyncCompile.warm_pool()
