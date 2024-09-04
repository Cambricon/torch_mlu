import os

import torch

torch._inductor.config.epilogue_fusion = False
torch._inductor.config.fallback_random = True
torch._inductor.config.assert_indirect_indexing = False
torch._inductor.config.max_autotune = True
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.inplace_buffers = False
torch._inductor.config.max_autotune_pointwise = True

torch._inductor.config.compile_threads = int(
    os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", 1)
)
torch._inductor.config.triton.max_tiles = 3
# torch._inductor.config.triton.use_block_ptr = True
torch._inductor.config.triton.persistent_reductions = False
torch._inductor.config.triton.max_block = {
    "X": 1048576,
    "Y": 1048576,
    "Z": 1048576,
    "R": 16384,
}

## used for debug
# torch._inductor.config.benchmark_kernel = True
torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.layout_optimization = 0
# torch._inductor.config.triton.debug_sync_kernel = True
# torch._inductor.config.triton.debug_sync_graph = True

from .fx_passes.mlu_post_pass import mlu_post_pass

torch._inductor.config.post_grad_custom_post_pass = mlu_post_pass
torch._inductor.config._save_config_ignore = {"post_grad_custom_post_pass"}
