import sympy
from typing import Tuple, List

import torch
from torch._inductor.dependencies import var_builder


def index_vars_no_squeeze(*argsizes: Tuple[sympy.Expr, ...], prefix: str):  # NOSONAR
    prefix = "w"
    var_ranges, add_var = var_builder(prefix)
    args: List[List[sympy.Symbol]] = []
    for size in argsizes:
        args.append(list(map(add_var, size)))
    return args, var_ranges


torch._inductor.dependencies.index_vars_no_squeeze = index_vars_no_squeeze
