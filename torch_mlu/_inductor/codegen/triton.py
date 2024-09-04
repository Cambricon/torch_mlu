import itertools
import collections
import functools
import sympy
import textwrap
import logging
from typing import Any, List, Set, Union, Counter, Dict, Optional, cast

import torch  # noqa
from torch._inductor import config, scheduler
from torch._inductor.virtualized import V, StoreMode
from torch._dynamo.utils import counters
from torch._inductor.dependencies import Dep, MemoryDep, StarDep, WeakDep
from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode
from torch._inductor.codecache import code_hash
from torch._inductor.codegen.multi_kernel import MultiKernel
from torch._inductor.codegen.common import (
    DeferredLine,
    CSEVariable,
    IndentedBuffer,
    SizeArg,
)
from torch._inductor.utils import (
    sympy_product,
    sympy_subs,
    sympy_dot,
    cache_on_self,
    next_power_of_2,
    Placeholder,
)
from torch._inductor.codegen.triton_utils import (
    signature_to_meta,
    signature_of,
    config_of,
)
from torch._inductor.codegen.triton import (
    TritonKernel,
    TritonScheduling,
    TritonOverrides,
    IndexingOptions,
    BlockPtrOptions,
    CandidateTiling,
    IterationRanges,
    IterationRangesEntry,
    IterationRangesRoot,
    EnableReduction,
    DisableReduction,
    TritonCSEVariable,
    triton_reshape,
    gen_common_triton_imports,
    perf_hint_log,
    texpr,
)

log = logging.getLogger(__name__)


@staticmethod
def maximum(a, b):
    return f"triton_helpers.maximum({b}, {a})"


TritonOverrides.maximum = maximum


# add a new patameter: is_pointwise_loop
def IterationRanges__init__(
    self,
    name: str,
    var_list: List[sympy.Symbol],
    var_ranges: Dict[sympy.Symbol, sympy.Expr],
    numel: sympy.Expr,
    prefix: str,
    *,
    kernel: TritonKernel,
    divisor=sympy.Integer(1),
    length=sympy.Integer(1),
    root: IterationRangesRoot,
    is_pointwise_loop: bool = False,
):
    # super().__init__()
    self.name = name
    self.var_list = var_list
    self.var_ranges = var_ranges
    self.numel = numel
    self.prefix = prefix
    self.divisor = divisor
    self.length = length
    self.kernel = kernel
    self.root = root
    self.is_pointwise_loop = is_pointwise_loop


torch._inductor.codegen.triton.IterationRanges.__init__ = IterationRanges__init__


def writeline(self, line):
    if self.root.is_loop or self.root.is_pointwise_loop:
        V.kernel.indexing_code.writeline(line)
    else:
        # lift non-reduction stores outside loop
        V.kernel.body.writeline(line)


torch._inductor.codegen.triton.IterationRangesEntry.writeline = writeline


def ranges_code(self, is_body=False):
    assert self.tensor_dim is not None
    size = self.kernel.indexing_size_str(self.tensor_dim)
    index_dtype = self.kernel.index_dtype
    convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
    if self.is_pointwise_loop and is_body:
        return f"tl.arange(0, {self.prefix.upper()}BLOCK_FRAGMENT){size}{convert}"
    else:
        return f"tl.arange(0, {self.prefix.upper()}BLOCK){size}{convert}"


torch._inductor.codegen.triton.IterationRangesRoot.ranges_code = ranges_code


def codegen_header(self, code):
    x = self.prefix
    if self.is_loop and self.prefix == "r" and self.grid_dim is None:
        code.writeline(f"{self.name} = {x}offset + {x}base")
        code.writeline(f"{x}mask = {self.name} < {x}numel")
    elif self.grid_dim is None:
        code.writeline(f"{self.name} = {self.ranges_code()}")
        code.writeline(f"{x}offset = 0")
        code.writeline(f"{x}mask = {self.name} < {x}numel")
    elif self.is_pointwise_loop:
        if self.tensor_dim is not None:
            line = f"{x}offset_begin + {self.ranges_code()}"
        else:
            line = self.scalar_code(f"{x}offset")
        code.writelines(
            [
                f"{x}offset_num = tl.cdiv({x.upper()}BLOCK, {x.upper()}BLOCK_FRAGMENT)",
                f"{x}step = tl.num_programs({self.grid_dim}) * {x.upper()}BLOCK_FRAGMENT",
                f"{x}offset_begin = {self.get_pid()} * {x.upper()}BLOCK_FRAGMENT",
            ]
        )
    else:
        if self.tensor_dim is not None:
            line = f"{x}offset + {self.ranges_code()}"
        else:
            line = self.scalar_code(f"{x}offset")
        code.writelines(
            [
                f"{x}offset = {self.get_pid()} * {x.upper()}BLOCK",
                f"{self.name} = {line}",
            ]
        )
        code.writeline(f"{x}mask = {self.name} < {x}numel")


torch._inductor.codegen.triton.IterationRangesRoot.codegen_header = codegen_header


def codegen_body(self, code):
    x = self.prefix
    if x == "z":
        code.writeline(
            f"{x}offset = offset // (yoffset_num * xoffset_num) * {x}step + {x}offset_begin"
        )
    elif x == "y":
        code.writeline(
            f"{x}offset = offset // xoffset_num % yoffset_num * {x}step + {x}offset_begin"
        )
    elif x == "x":
        code.writeline(f"{x}offset = offset % xoffset_num * {x}step + {x}offset_begin")

    line = f"{x}offset + {self.ranges_code(True)}"
    code.writeline(f"{self.name} = {line}")
    code.writeline(f"{x}mask = {self.name} < {x}numel")


torch._inductor.codegen.triton.IterationRangesRoot.codegen_body = codegen_body


def IterationRangesRoot__init__(
    self,
    name: str,
    numel: sympy.Expr,
    prefix: str,
    index: int,
    kernel: TritonKernel,
    pid_cache=None,
    *,
    is_loop: bool,
    tensor_dim: Optional[int],
    grid_dim: Optional[int],
    is_pointwise_loop: bool = False,
):
    if pid_cache is None:
        pid_cache = {}
    IterationRanges.__init__(
        self,
        name=name,
        var_list=[],
        var_ranges={},
        numel=numel,
        prefix=prefix,
        kernel=kernel,
        root=self,
        is_pointwise_loop=is_pointwise_loop,
    )
    self.index = index
    # Store all the nodes in one flat list
    self.nodes: Dict[sympy.Expr, IterationRangesEntry] = {}
    # This is for re-ordering program ID in triton mm template
    # pid_cache["tl.program_id(0)"] = pid_m
    self.pid_cache: Dict[str, str] = pid_cache

    # True if the dimension is implemented as a single program looping over
    # the full dimension (currently only used for non-persistent reduction)
    self.is_loop = is_loop
    self.is_pointwise_loop = is_pointwise_loop
    # Index of corresponding dimension on triton tensors
    self.tensor_dim = tensor_dim
    # Index of corresponding dimension in the triton grid
    self.grid_dim = grid_dim


torch._inductor.codegen.triton.IterationRangesRoot.__init__ = (
    IterationRangesRoot__init__
)


@staticmethod
def create(
    strides: List[sympy.Expr],
    constant_offset: sympy.Expr,
    range_trees: List[IterationRangesEntry],
    mask_vars: Set[sympy.Symbol],
) -> BlockPtrOptions:
    """Helper to create a  BlockPtrOptions instance"""
    block_shape = [
        f"{t.prefix.upper()}BLOCK_FRAGMENT"
        if t.is_pointwise_loop
        else f"{t.prefix.upper()}BLOCK"
        for t in range_trees
    ]

    reshape_suffix = [*block_shape]

    broadcasting_dim = [s == 0 for s in strides]
    for i, is_broadcasting in enumerate(broadcasting_dim):
        if is_broadcasting:
            # drop any stride==0 dimensions for performance
            reshape_suffix[i] = "1"

    if V.kernel.no_x_dim:
        assert range_trees[0].prefix == "x"
        reshape_suffix.pop(0)

    if (
        not V.kernel.inside_reduction
        and len(strides) == len(V.kernel.numels) - 1
        and V.kernel.numels[-1] != 1
    ):
        # Need to expand rank by 1 to match rank when self.inside_reduction=True
        reshape_suffix.append("1")

    def filter(it):
        """Removes any broadcasting dims from a given sequence"""
        assert len(it) == len(broadcasting_dim)
        return [
            item
            for item, is_broadcasting in zip(it, broadcasting_dim)
            if not is_broadcasting
        ]

    return BlockPtrOptions(
        constant_offset=V.graph.sizevars.lookup_precomputed_size(constant_offset),
        shape=[
            V.graph.sizevars.lookup_precomputed_size(t.numel)
            for t in filter(range_trees)
        ],
        strides=[*map(V.graph.sizevars.lookup_precomputed_size, filter(strides))],
        block_shape=filter(block_shape),
        order=V.graph.sizevars.guarded_order(filter(strides)),
        offsets=filter([f"{t.prefix}offset" for t in range_trees]),
        mask_vars=mask_vars,
        reshape_suffix=reshape_suffix,
    )


torch._inductor.codegen.triton.BlockPtrOptions.create = create


@cache_on_self
def boundary_check(self) -> List[int]:
    """List of indices to pass to tl.load(boundary_check=...)"""
    check = []
    for i in range(len(self.shape)):
        if (
            self.block_shape[i] != "1"
            and not V.graph.sizevars.statically_known_equals(self.strides[i], 0)  # type: ignore[arg-type]
            # Modified by Cambricon: comment below four lines:
            # and not V.graph.sizevars.statically_known_multiple_of(
            #    self.shape[i],
            #    config.triton.max_block[self.block_shape[i][0]],  # type: ignore[arg-type]
            # )
            # Cambricon modified end.
            and not (V.kernel.no_x_dim and self.block_shape[i] == "XBLOCK")
        ):
            check.append(i)
    return check


torch._inductor.codegen.triton.BlockPtrOptions.boundary_check = boundary_check


class MluTritonKernel(TritonKernel):
    def initialize_range_tree(self, pid_cache):
        no_r_dim = not self.inside_reduction or self.numels[-1] == 1
        enable_pointwise_loop = self.numels[-1] == 1

        prefixes = "zyxr"
        active_prefixes = prefixes[-len(self.numels) :]

        grid_dims = "xyz"
        if self.no_x_dim:
            tensor_dims = "r"
        elif no_r_dim:
            tensor_dims = "xyz"
        else:
            tensor_dims = "xyzr"

        tensor_dims = "".join(p for p in tensor_dims if p in active_prefixes)

        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix == "r"
            tensor_dim = tensor_dims.find(prefix) if prefix in tensor_dims else None
            grid_dim = None if is_reduction else grid_dims.find(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    self.numels[i],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                    is_pointwise_loop=enable_pointwise_loop and not is_reduction,
                )
            )
        for tree in self.range_trees:
            # reduction indexing goes inside a loop
            # if not (tree.is_loop or tree.is_pointwise_loop):
            if not tree.is_loop:
                tree.codegen_header(self.body)
        if self.inside_reduction and self.range_trees[-1].is_loop:
            # workaround for this issue:
            # https://gist.github.com/jansel/6527126f781559095c5531f98a4235a7
            self.body.writeline(f"rbase = {self.range_trees[-1].ranges_code()}")

    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=None,
        dense_indexing=False,
        override_mask=None,
        block_ptr=False,
    ) -> Union[IndexingOptions, BlockPtrOptions]:
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.simplify_indexing(index)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        # if simple replacements didn't get rid of floor/ceil, try full subs
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        # last resort, if no range vars are in the expr, hoist it
        # TODO instead of trying to blindly find complicated exprs, we should hoist the
        # inputs/outputs sizes and strides, but at the time indexing is generated
        # kernel inputs and outputs are not set yet, we'd need a deeper refactor
        # to do it this way

        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                # for nested exprs, atoms yields top level first (?)
                # so if everything goes fine, lower level replacements will come up empty
                symbols = a.free_symbols
                if len(symbols) > 0 and all(
                    s.name.startswith("s") or s.name.startswith("ps") for s in symbols
                ):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)

        index = self.simplify_indexing(index)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars: Set[str] = set()
        for var in index_vars:
            assert isinstance(var, sympy.Symbol)
            has_rindex = has_rindex or var.name.startswith("r")
            if override_mask:
                pass
            elif var.name.startswith("tmp"):
                # indirect indexing
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif var.name.startswith(("s", "ps", "i", "u")):
                pass
            else:
                # Modified by Cambricon start: replace with new codes
                # The default value for max_tiles is 2, we support max_tiles = 3.
                # so, we add a new dimension.
                # var is one of xN, yN, zN or rN
                assert var.name[0] in "xyzr", var.name
                # Original codes:
                # var is one of xN, yN or rN
                # assert var.name[0] in "xyr", var.name
                # Modified by Cambricon end

                mask_vars.add(f"{var.name[0]}mask")

        need_dense = (
            config.triton.dense_indexing
            or dense_indexing
            or self._load_mask is not None
        ) and index != 0

        have_dense = True
        have_loop_vars = False
        dense_mask_vars = set()

        for tree in self.active_range_trees():
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f"{tree.prefix}mask")

        if (
            block_ptr
            and config.triton.use_block_ptr
            and not override_mask
            and not self._load_mask
            and len(mask_vars - dense_mask_vars) == 0
            and not self.is_indirect_indexing(index)
            and have_loop_vars
            # workaround https://github.com/openai/triton/issues/2821
            and self.index_dtype == "tl.int32"
        ):
            index_relative_to_xyr_index = sympy_subs(
                index, {v: t.expr for v, t in self.range_tree_nodes.items()}
            )
            range_trees = self.active_range_trees(reorder=True)
            # range_trees = self.active_range_trees(reorder=False)
            symbols = [t.symbol() for t in range_trees]
            strides = [sympy.Wild(f"stride_{s}", exclude=symbols) for s in symbols]
            offset = sympy.Wild("_offset", exclude=symbols)
            m = index_relative_to_xyr_index.match(sympy_dot(symbols, strides) + offset)
            # TODO(jansel): it is sometimes possible to do higher dimensional block_ptrs with
            #               a tl.reshape the correct block.  We will miss these cases today.
            if m:
                self.filter_masks(mask_vars)
                return BlockPtrOptions.create(
                    [m[s] for s in strides],
                    m[offset],
                    range_trees,
                    mask_vars,  # type: ignore[arg-type]
                )

        expand_str = None
        index_str = self.index_to_str(index)
        if isinstance(index, sympy.Integer):
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
            return IndexingOptions(index_str, set(), "None", expand_str, has_rindex)

        if need_dense and not have_dense:
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            index_str = f"tl.broadcast_to({index_str}, {expand_str})"
            mask_vars = dense_mask_vars
        elif not have_loop_vars and copy_shape:
            index_str = f"tl.broadcast_to({index_str}, {copy_shape}.shape)"
            mask_vars = dense_mask_vars

        if override_mask:
            mask_vars = {override_mask}

        if self._load_mask:
            mask_vars.add(self._load_mask)

        self.filter_masks(mask_vars)

        mask_str = " & ".join(sorted(map(str, mask_vars))) if mask_vars else "None"
        return IndexingOptions(index_str, mask_vars, mask_str, expand_str, has_rindex)  # type: ignore[arg-type]

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        original_index = index
        indexing = self.indexing(index, dense_indexing=True, block_ptr=mode is None)

        # Guard against write-after-read corruption in triton.
        # See # https://github.com/openai/triton/issues/1615
        # This triton bug means that a load which is broadcasted over multiple
        # warps may see the result of a store that happens later in the triton
        # program. The workaround is to add a barrier before storing, which
        # enforces that all warps have already read the data.
        is_inplace = name in self.args.inplace_buffers
        is_broadcasted = self.is_broadcasted(original_index)
        # Modified by Cambricon, start: comment below codes
        # if is_inplace and is_broadcasted:
        #     self.stores.writeline(DeferredLine(name, "tl.debug_barrier()"))
        # Modified by Cambricon, end

        advance_block_ptr = None
        if isinstance(indexing, BlockPtrOptions):
            block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                name, var, indexing
            )
            # block_ptr stores don't do implicit casting
            line = self.codegen_block_ptr_store_line(
                name, indexing, block_ptr, value, other
            )
        elif mode is None:
            line = f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})"
        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + ({indexing.index_str}), {value}, {indexing.mask_str})"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(DeferredLine(name, line))
        if advance_block_ptr:
            self.stores.writeline(advance_block_ptr)

        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def filter_masks(self, mask_vars):
        pass

    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.suffix
        ):
            return

        if self.inside_reduction and self.range_trees[-1].is_loop:
            # self.body.writeline("for zoffset in range(0, znumel, ZBLOCK):")
            self.body.writeline("for roffset in range(0, rnumel, RBLOCK):")
            with self.body.indent():
                # last range tree is always reduction
                self.range_trees[-1].codegen_header(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)

            # invalidate any caches that came from inside the reduction loop
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
        elif self.range_trees[0].is_pointwise_loop:
            if len(self.range_trees) - 1 == 3:
                self.body.writeline(
                    "for offset in range(zoffset_num * yoffset_num * xoffset_num):"
                )
            elif len(self.range_trees) - 1 == 2:
                self.body.writeline("for offset in range(yoffset_num * xoffset_num):")
            elif len(self.range_trees) - 1 == 1:
                self.body.writeline("for offset in range(xoffset_num):")
            else:
                print("EEEEEEE")
            with self.body.indent():
                for idx in range(len(self.range_trees) - 1):
                    self.range_trees[idx].codegen_body(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)
        else:
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        size_hints = []
        for numel in self.numels:
            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                # This default heuristic hint was picked carefully: it is
                # large, to ensure that we don't shrink the block size (since
                # if you don't have many elements, it'd be wasteful to pick a
                # large block size).  Since we don't know how many elements we
                # might have, we should be OK with some inefficiency to make
                # sure we handle the large case well.  8192 is the largest
                # block size we support, so we pick that.
                #
                # If we have a better hint for unbacked SymInts (e.g., because
                # a user told us, or we are tracking upper bounds) we could
                # use that here.
                size_hint = 8192
            else:
                # For mlu, it is better not to do next_power_of_2.
                # size_hint = next_power_of_2(int(numel_hint))
                size_hint = int(numel_hint)
            size_hints.append(size_hint)

        if not self.inside_reduction:
            size_hints.pop()

        heuristics = self._get_heuristic()

        if name is None:
            code.splice(gen_common_triton_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature = self.args.python_argdefs()
        # maps actual expression to SizeArg if it is in sizevars replacements
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                # mypy is unhappy about the sympy.Expr
                # type for the key of the dict below
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype
        )
        triton_meta = {
            "signature": triton_meta_signature,
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
        }

        inductor_meta = {
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "no_x_dim": self.no_x_dim,
            "backend_hash": torch.utils._triton.triton_hash_with_backend(),
        }
        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            triton_meta_signature[len(argdefs)] = signature_of(
                sizearg, size_dtype=self.index_dtype
            )
            argdefs.append(f"{tree.prefix}numel")
            # constexpr version causes issues, see
            # https://github.com/pytorch/torchdynamo/pull/1362
            # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
            #     tree.numel
            # )
            # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/openai/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
            triton_meta["constants"][arg_num] = 1  # type: ignore[index]

        self.triton_meta = triton_meta

        for tree in self.range_trees:
            if tree.prefix == "r" and self.persistent_reduction:
                # RBLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue
            argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")
            if tree.prefix != "r" and tree.is_pointwise_loop:
                argdefs.append(f"{tree.prefix.upper()}BLOCK_FRAGMENT : tl.constexpr")

        self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()

    def dense_size_list(self) -> List[str]:
        sizes = ["1"] * self.triton_tensor_ndim()
        for tree in self.range_trees:
            if tree.tensor_dim is None:
                continue

            if tree.is_pointwise_loop:
                sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK_FRAGMENT"
            else:
                sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"

        return sizes

    def imports_for_benchmark_kernel(self):
        return textwrap.dedent(
            """
            import torch
            import torch_mlu
            from torch._dynamo.testing import rand_strided
            {}
            from torch._inductor.triton_heuristics import grid, split_scan_grid
        """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )


class MluTritonScheduling(TritonScheduling):
    def codegen_sync(self):
        V.graph.wrapper_code.writeline("torch.mlu.synchronize()")

    # Significantly different from the community implementation, we have modified
    # it to return multi-dimensional tilings.
    @staticmethod
    @functools.lru_cache(32)
    def candidate_tilings(node):
        tilings = TritonScheduling.candidate_tilings(node)

        ranges, reduction_ranges = node.get_ranges()
        if len(ranges) <= 1:
            return tilings

        rw = node.pointwise_read_writes()

        # isinstance(dep, MemoryDep): this filters out StarDeps. StarDeps refer to reads
        # that need to access the entire tensor; they don't contribute read indexing
        # information (and practically, they don't have dep.index so they can't be used
        # for stride_hints below
        dep_sources = [rw.reads, rw.writes]
        deps = [
            dep
            for dep in itertools.chain.from_iterable(dep_sources)
            if dep.name not in V.graph.removed_buffers and isinstance(dep, MemoryDep)
        ]
        write_names = {dep.name for dep in rw.writes}

        for dep in deps:
            strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
            assert len(strides) == len(ranges)
            splits = [0]
            previous_is_zero = strides[0] == 0
            for i in range(1, len(strides)):
                current_is_zero = strides[i] == 0
                if previous_is_zero != current_is_zero:
                    splits.append(i)
                    previous_is_zero = current_is_zero
            if len(splits) == 1:
                continue
            splits.append(len(strides))
            if len(splits) > 4:
                splits = splits[:3] + splits[-1:]
            tiled_groups = tuple(
                V.graph.sizevars.simplify(
                    sympy_product(ranges[splits[i] : splits[i + 1]])
                )
                for i in range(len(splits) - 1)
            )

            # score by number of elements
            score = V.graph.sizevars.size_hint(sympy_product(ranges))
            if dep.name in write_names:
                # ngimel said contiguous writes is more important than reads
                score *= 2
            score *= len(splits)

            if (
                V.graph.sizevars.size_hint(
                    score - sympy_product(itertools.chain(ranges, reduction_ranges))
                )
                >= 0
            ):
                tilings.append(CandidateTiling(tiled_groups, score, dep.name))
        return tuple(tilings)

    def codegen_node_schedule(
        self, node_schedule, buf_accesses, numel, reduction_numel
    ):
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel

        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
            node_schedule, numel, reduction_numel
        )

        is_split_scan = any(
            isinstance(node, BaseSchedulerNode) and node.is_split_scan()
            for node in node_schedule
        )
        # Modified by Cambricon: replace TritonKernel with MluTritonKernel
        kernel_type = TritonSplitScanKernel if is_split_scan else MluTritonKernel
        # original:
        # kernel_type = TritonSplitScanKernel if is_split_scan else TritonKernel
        # Modified by Cambricon, end
        kernel_args = tiled_groups
        kernel_kwargs = {
            "reduction_hint": reduction_hint_val,
            "mutations": mutations,
            "index_dtype": index_dtype,
        }
        kernel = kernel_type(
            *kernel_args,
            **kernel_kwargs,
        )
        kernel.buf_accesses = buf_accesses

        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()

        kernel_name = self.define_kernel(src_code, node_schedule)
        log.debug("Generating kernel code with kernel_name: %s", kernel_name)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        if kernel.persistent_reduction and config.triton.multi_kernel:
            kernel2 = TritonKernel(
                *kernel_args,
                **kernel_kwargs,
                disable_persistent_reduction=True,
            )
            self.codegen_node_schedule_with_kernel(node_schedule, kernel2)
            with V.set_kernel_handler(kernel2):
                src_code2 = kernel2.codegen_kernel()
            kernel_name2 = self.define_kernel(src_code2, node_schedule)
            kernel2.kernel_name = kernel_name2
            kernel2.code_hash = code_hash(src_code2)

            final_kernel = MultiKernel([kernel, kernel2])
        else:
            final_kernel = kernel  # type: ignore[assignment]

        with V.set_kernel_handler(final_kernel):
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()

        self.codegen_comment(node_schedule)
        final_kernel.call_kernel(final_kernel.kernel_name)
        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
            V.graph.wrapper_code.supports_intermediate_hooks
            and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernel.args.live_output_buffers()
            for node in node_schedule:
                if not isinstance(node, scheduler.BaseSchedulerNode):
                    continue
                name = node.get_name()
                if name not in live_outs:
                    continue
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
        if reduction_numel != 1 or config.triton.max_tiles <= 1:
            # TODO(jansel): should we tile reductions?
            # do perf hint here if stride-1 dim is not being reduced
            if perf_hint_log.level <= logging.WARNING:
                for node in EnableReduction.filter(node_schedule):
                    if len(cls.candidate_tilings(node)) > 0:
                        perf_hint_log.info("reduction over non-contiguous dims")
                        break
            return (numel, reduction_numel)

        seen_names = set()
        candidate_tiles: Counter[Any] = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for tiling in cls.candidate_tilings(node):
                if tiling.name in seen_names:
                    continue
                seen_names.add(tiling.name)
                candidate_tiles[tiling.tiling] += tiling.score

        ranked_tilings = [tiling for tiling, score in candidate_tiles.most_common()]

        # Modified by Cambricon: comment below lines
        # Reason: we does not support max_tiles > 3 for now, so disable below.
        # if config.triton.max_tiles >= 3:
        #     # Consider adding a third dimension of tiling, but only
        #     # when a1 is a multiple of b1; otherwise, you have a lot
        #     # of stragglers which is annoying to generate code for.
        #     #
        #     # NB: More than three max tiles is not enabled by default.

        #     # Add one 3D tiling choice
        #     for i in range(1, len(ranked_tilings)):
        #         a0, a1 = ranked_tilings[0]
        #         b0, b1 = ranked_tilings[i]
        #         if V.graph.sizevars.size_hint(a1 - b1) == 0:
        #             continue
        #         if V.graph.sizevars.size_hint(a1 - b1) < 0:
        #             # swap so a0 is bigger
        #             a0, a1 = ranked_tilings[i]
        #             b0, b1 = ranked_tilings[0]
        #         assert V.graph.sizevars.size_hint(a1 - b1) > 0
        #         if V.graph.sizevars.statically_known_multiple_of(a1, b1):
        #             tiling = (a0, FloorDiv(a1, b1), b1)
        #             ranked_tilings = [tiling] + ranked_tilings
        #             break  # only 1 choice for now
        # Modified by Cambricon end

        if len(ranked_tilings) > 1:
            perf_hint_log.info("possibly bad tiling: %s", ranked_tilings)

        for tiled_groups in ranked_tilings:
            new_groups = (*tiled_groups, reduction_numel)
            if all(
                TritonKernel.is_compatible(new_groups, node.get_ranges())
                for node in node_schedule
                if isinstance(node, scheduler.SchedulerNode)
            ):
                return new_groups

        return (numel, reduction_numel)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        indexing = self.indexing(index, block_ptr=True)
        has_rindex = indexing.has_rindex()
        has_tmpmask = indexing.has_tmpmask()

        # Keep the variable in cache if were going to reuse it. Equiv., if any of the following hold
        #  1) We are doing broadcasting
        #  2) It is a non-coalesced load. The intuition is that if it's
        #  non-coalesced, we will likely load each element multiple times in
        #  practice.
        #  3) It will be used later and it won't be CSE'd. Equiv., if all the following hold
        #   3.1) We are in a reduction loop
        #   3.2) Its not its last use
        #   3.3) This load will not be lifted to the body
        #
        is_coalesced = any(
            i == 1 for i in self.get_strides_of_load(original_index).values()
        )
        if self.is_broadcasted(original_index):
            ep = ", eviction_policy='evict_last'"
        elif not is_coalesced:
            ep = ", eviction_policy='evict_last'"
        elif self.inside_reduction and self.range_trees[-1].is_loop:
            if name in self.args.inplace_buffers:
                names = set(self.args.inplace_buffers[name].other_names)
            else:
                names = {name}
            last_use = len(names & self.last_usage) > 0
            evict_last = not last_use and (has_rindex or indirect_indexing)
            if evict_last:
                ep = ", eviction_policy='evict_last'"
            else:
                ep = ", eviction_policy='evict_first'"
        elif self.range_trees[-1].is_pointwise_loop:
            if name in self.args.inplace_buffers:
                names = set(self.args.inplace_buffers[name].other_names)
            else:
                names = {name}
            last_use = len(names & self.last_usage) > 0
            evict_last = not last_use and (has_rindex or indirect_indexing)
            if evict_last:
                ep = ", eviction_policy='evict_last'"
            else:
                ep = ", eviction_policy='evict_first'"
        else:
            ep = ""
        # "other" below is a workaround for https://github.com/openai/triton/issues/737
        # for bool, even though it's likely subject to the same bug, setting `other` leads
        # to LLVM errors so we are skipping it for now
        if (
            (has_tmpmask or has_rindex)
            and V.graph.get_dtype(name) != torch.bool
            and indexing.has_mask()
        ):
            other = ", other=0.0"
        else:
            other = ""

        advance_block_ptr = None
        append_broadcast = None
        if V.graph.is_unspec_arg(name):
            line = var
        else:
            if isinstance(indexing, BlockPtrOptions):
                block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                    name, var, indexing, other
                )
                line = f"tl.load({block_ptr}{other}{ep})"
                # add needed size=1 dimensions
                line = triton_reshape(
                    line, indexing.block_shape, indexing.reshape_suffix
                )
            elif isinstance(original_index, sympy.Integer):
                line = f"tl.load({var} + ({original_index}))"
                append_broadcast = indexing.expand_str
            else:
                line = f"tl.load({var} + ({indexing.index_str}), {indexing.mask_str}{ep}{other})"

            dtype = V.graph.get_dtype(name)
            if dtype in (torch.float16, torch.bfloat16):
                line += ".to(tl.float32)"
            if dtype == torch.bool and torch.version.hip is None:
                # Workaround for https://github.com/openai/triton/issues/2151
                # tl.load returns int8 when loading from pointer to int1
                # NOTE: Currently causes hangs on bool UTs for ROCm
                line += ".to(tl.int1)"

        if has_tmpmask:
            # Masked loads must come after the mask is computed
            load_buffer = self.compute
        elif (
            self.inside_reduction
            and self.range_trees[-1].is_loop
            and not indirect_indexing
            and not has_rindex
        ):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            load_buffer = self.body
        else:
            load_buffer = self.loads

        result_var = self.cse.generate(load_buffer, line)
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

        if append_broadcast:
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            result_var = self.cse.generate(load_buffer, line)

        if advance_block_ptr:
            load_buffer.writeline(advance_block_ptr)

        if not self.inside_reduction or (not indexing.has_rmask() and not has_rindex):
            self.outside_loop_vars.add(result_var)

        return result_var
