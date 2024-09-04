# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import re
from collections import namedtuple

PLUGIN_NAME = 'pytorch_profiler'

WORKER_PATTERN = re.compile(r"""^(.*?) # worker name
        (\.\d+)? # optional timestamp like 1619499959628 used as span name
        \.pt\.trace\.json # the ending suffix
        (?:\.gz)?$""", re.X)  # optional .gz extension

NODE_PROCESS_PATTERN = re.compile(r"""^(.*)_(\d+)""")
MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS = 10
MAX_MLU_PER_NODE = 64

View = namedtuple('View', 'id, name, display_name')
OVERALL_VIEW = View(1, 'overall', 'Overview')
OP_VIEW = View(2, 'operator', 'Operator')
KERNEL_VIEW = View(3, 'kernel', 'Kernel')
TRACE_VIEW = View(4, 'trace', 'Trace')
DISTRIBUTED_VIEW = View(5, 'distributed', 'Distributed')
MEMORY_VIEW = View(6, 'memory', 'Memory')
MODULE_VIEW = View(7, 'module', 'Module')
LIGHTNING_VIEW = View(8, 'lightning', 'Lightning')

TOOLTIP_MLU_UTIL = \
    'MLU Utilization:\n' \
    'MLU busy time / All steps time. The higher, the better. ' \
    'MLU busy time is the time during which there is at least one MLU kernel running on it. ' \
    'All steps time is the total time of all profiler steps(or called as iterations).\n'
TOOLTIP_TENSOR_CORES = \
    'Kernel using Tensor Cores:\n' \
    'Total MLU Time for Tensor Core kernels / Total MLU Time for all kernels.\n' \
    'It is always 0.0 On MLU devices because MLU does not using Tensor Cores.\n'
TOOLTIP_OCCUPANCY_TABLE = \
    "This \"Mean\" number is the weighted average of all calls' OCC_K of the kernel, " \
    "using each call's execution duration as weight. " \
    'It shows fine-grained low-level MLU utilization.'
TOOLTIP_BLOCKS_PER_SM = \
    'Blocks Per SM = blocks of this kernel / SM number of this MLU.\n' \
    'If this number is less than 1, it indicates the MLU multiprocessors are not fully utilized.\n' \
    '\"Mean Blocks per SM\" is the weighted average of all calls of this kernel, ' \
    "using each call's execution duration as weight."
TOOLTIP_OP_TC_ELIGIBLE = \
    'Whether this operator is eligible to use Tensor Cores.'
TOOLTIP_OP_TC_SELF = \
    'Time of self-kernels with Tensor Cores / Time of self-kernels.'
TOOLTIP_OP_TC_TOTAL = \
    'Time of kernels with Tensor Cores / Time of kernels.'
TOOLTIP_KERNEL_USES_TC = \
    "Whether this kernel uses Tensor Cores. It's always NO On MLU devices because MLU does not have Tensor Cores."
TOOLTIP_KERNEL_OP_TC_ELIGIBLE = \
    'Whether the operator launched this kernel is eligible to use Tensor Cores.'
