import contextlib
import functools
import gc
import itertools
import threading
import warnings
import weakref
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import torch
import torch_mlu
from torch import Tensor
from torch.types import _bool
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.cudagraph_trees import (
    AliasesPriorGraphOutput,
    AliasesNewOutput,
    CompilationMode,
    CUDAGraphNode,
    CUDAGraphTreeManager,
    GraphID,
    ExecutionState,
    FunctionID,
    OutputAliasInfo,
    StorageWeakRefWrapper,
    UnaliasedStorage,
    WrappedFunction,
    clear_cublas_manager,
    disable_conv_cache_emptying,
    format_tb,
    get_container,
    get_history_recording,
    is_live,
    map_to_ref,
)
from torch._inductor.compile_fx import (
    get_expanded_dims,
    static_input,
)
from torch.utils.weak import TensorWeakRef
from torch._inductor import config

StorageDataPtr = int

# A path index of (depth, offset) indices into a graph that is `depth`` number of nodes from the root
# at graph output offset
PathOutputIndex = Tuple[int, int]

# Aliases for List that say what the indices denote
InputList = List  # input indexes
OutputList = List  # output indexes
LevelList = List  # levels (distance from root of tree)

StackTraces = List[Optional[str]]

if torch.backends.mlu.is_built():
    from torch_mlu._MLUC import (
        _mlu_MLUAllocator_AllocatorState as AllocatorState,
        _set_cached_tensors_enabled as _set_cached_tensors_enabled,
    )
else:

    class AllocatorState:  # type: ignore[no-redef]
        pass

    def _set_cached_tensors_enabled(enabled: _bool) -> None:
        pass


log = torch._logging.getArtifactLogger(__name__, "cudagraphs")


@contextlib.contextmanager
def enable_history_recording():
    "Turns on history recording in the CUDA Caching Allocator"
    enabled = torch_mlu._MLUC._mlu_isHistoryEnabled()
    try:
        if not enabled:
            torch.mlu.memory._record_memory_history()
        yield
    finally:
        if not enabled:
            torch.mlu.memory._record_memory_history(None)


torch._inductor.cudagraph_trees.enable_history_recording = enable_history_recording


def TreeManagerContainer__init__(self, device_index):
    # This class keeps a strong reference to tree_manager,
    # but upon all other strong references to the tree_manager will reset it to None.
    # We need a strong reference so that we can still access its attributes upon cleanup.
    self.tree_manager: Optional[CUDAGraphTreeManager] = None

    # Number of outstanding references to the current tree manager
    self.live_cudagraphify_fns = 0

    self.device_index = device_index

    # Following two objects are only set in the case that Tensor outputs outlive
    # the cudagraphify_fns. Reference to the Graph is needed to keep the private pool from
    # deallocation.
    self.live_storages_count = 0
    self.graph: Optional[torch.mlu.MLUGraph] = None

    self.lock = threading.Lock()


torch._inductor.cudagraph_trees.TreeManagerContainer.__init__ = (
    TreeManagerContainer__init__
)


def StorageWeakRefWrapper_expired(self):
    if self.extra_ref_check is not None and not self.extra_ref_check():
        return False

    # if extra_ref_check is not None we expect an additional reference
    stor_count = torch_mlu._MLUC._storage_Use_Count(self.ref.cdata)
    return (stor_count - (self.extra_ref_check is not None)) == 0


StorageWeakRefWrapper.expired = StorageWeakRefWrapper_expired


@contextlib.contextmanager
def _use_cuda_memory_pool_manager(device, mem_pool, stream):
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
    torch.mlu.synchronize()
    stream.wait_stream(torch.mlu.current_stream())

    with torch.mlu.stream(stream), torch.device(device):
        torch_mlu._MLUC._mlu_beginAllocateCurrentStreamToPool(device, mem_pool)
        try:
            yield
        finally:
            torch_mlu._MLUC._mlu_endAllocateCurrentStreamToPool(device, mem_pool)
            torch_mlu._MLUC._mlu_releasePool(device, mem_pool)

    torch.mlu.current_stream().wait_stream(stream)


torch._inductor.cudagraph_trees._use_cuda_memory_pool_manager = (
    _use_cuda_memory_pool_manager
)


def CUDAWarmupNode__init__(
    self,
    wrapped_function: WrappedFunction,
    parent,
    cuda_graphs_pool: Tuple[int, int],
    existing_cuda_graph: Optional[torch.mlu.MLUGraph],
    device_index: int,
    stack_traces: Optional[StackTraces],
    stream: torch.mlu.Stream,
    already_warm: bool,
):
    self.wrapped_function = wrapped_function
    self.parent = parent
    self.cuda_graphs_pool = cuda_graphs_pool
    self.outputs_weakrefs: List[Optional[StorageWeakRefWrapper]] = []
    self.tensor_weakrefs: List[Optional[TensorWeakRef]] = []
    self.existing_cuda_graph = existing_cuda_graph
    self.has_run = False
    self.device_index = device_index
    self.stack_traces = stack_traces
    self.stream = stream
    self.already_warm = already_warm


torch._inductor.cudagraph_trees.CUDAWarmupNode.__init__ = CUDAWarmupNode__init__


def CUDAWarmupNode_run(self, new_inputs):
    assert not self.has_run, "Wrapped function should never be run twice"

    # See: output_is_alias_of_persistent_static_inputs below. We should only be returning freshly created
    # storages in path_live_weakrefs.
    existing_path_data_ptrs = {t.data_ptr() for t in self.path_live_weakrefs() if t()}

    def get_non_cudagraph_inps():
        non_cudagraph_inps = set()
        for t in itertools.chain(new_inputs, self.wrapped_function.constants):
            if (
                isinstance(t, torch.Tensor)
                and t.untyped_storage().data_ptr() not in existing_path_data_ptrs
            ):
                non_cudagraph_inps.add(t.untyped_storage().data_ptr())
        return non_cudagraph_inps

    non_cudagraph_inps = get_non_cudagraph_inps()

    if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
        refs = list(self.path_live_weakrefs())
        check_memory_pool(self.device_index, self.cuda_graphs_pool, refs)

    with torch.mlu.device(self.device_index), _use_cuda_memory_pool_manager(
        self.device_index, self.cuda_graphs_pool, self.stream
    ), get_history_recording():
        out = self.wrapped_function.model(new_inputs)

    assert len(new_inputs) == 0

    # sdpa returns cpu tensors when not recording cuda graph
    def add_ref(o):
        return (
            o is not None
            and isinstance(o, torch.Tensor)
            and o.is_mlu
            and o.untyped_storage().data_ptr() not in non_cudagraph_inps
            and o.untyped_storage().data_ptr() != 0
        )

    self.outputs_weakrefs.extend([map_to_ref(o) if add_ref(o) else None for o in out])
    self.tensor_weakrefs.extend([TensorWeakRef(o) if add_ref(o) else None for o in out])

    if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
        out_refs = self.path_live_weakrefs()
        new_storages = [t for t in out_refs if t.data_ptr() not in non_cudagraph_inps]
        check_memory_pool(self.device_index, self.cuda_graphs_pool, new_storages)

    return out


torch._inductor.cudagraph_trees.CUDAWarmupNode.run = CUDAWarmupNode_run


def CUDAGraphNode__init__(
    self,
    wrapped_function: WrappedFunction,
    id: GraphID,
    parent: Optional[CUDAGraphNode],
    inputs: List[Tensor],
    cuda_graphs_pool: Tuple[int, int],
    device_index: int,
    stack_traces: Optional[StackTraces],
    stream: torch.mlu.Stream,
):
    assert isinstance(inputs, (list, tuple))

    self.wrapped_function = wrapped_function
    self.id = id
    self.device = device_index
    self.stack_traces = stack_traces
    self.stream = stream

    # if this is a root parent will be None. use weakref to prevent reference cycle
    self._parent = weakref.ref(parent) if parent is not None else None
    # reference to the shared memory pool for the entire cuda graphs tree
    self.cuda_graphs_pool = cuda_graphs_pool

    # A single wrapped function may be recorded multiple times if memory patterns or
    # invariants change from one execution to the next
    self.children: Dict[FunctionID, List[CUDAGraphNode]] = defaultdict(list)

    # StorageWeakRef maintains whether the Storage C++ object remains allocated,
    # not whether the corresponding memory has been deallocated. In order
    # to use them to track memory deallocations we must maintain a single StorageWeakRef
    # for all Storages that reference that memory (even if we are constructing Storages
    # that do not have a deallocator function). We maintain one single storage_cache
    # as we execute any tree path. When we retrieve a storage from the cache we
    # check that it is still alive, and we hash based on observed recording data ptr
    # and storage cdata.

    # we preserve a single reference to executed outputs that is then referenced
    # in children to avoid children having to chase parent pointers in the hot path
    # DO NOT reassign output_weakrefs, only call `clear()`
    # Path is a series of nodes from root to the current node
    self.outputs_weakrefs: OutputList[Optional[StorageWeakRefWrapper]] = []
    self.path_weakrefs: LevelList[OutputList[Optional[StorageWeakRefWrapper]]] = [
        node.outputs_weakrefs for node in self._path_from_root
    ]
    self.path_stacktraces: LevelList[StackTraces] = [
        node.stack_traces for node in self._path_from_root
    ]
    self.tensor_weakrefs: OutputList[Optional[TensorWeakRef]] = []

    # tensors which are outputs of previous graphs in the tree
    self.cudagraph_managed_idxs: List[int] = [
        idx
        for idx, t in enumerate(inputs)
        if isinstance(t, torch.Tensor) and self._is_cuda_graph_recorded_tensor(t)
    ]

    self.static_input_idxs: List[int] = list(
        set(wrapped_function.static_input_idxs) | set(self.cudagraph_managed_idxs)
    )

    self.static_input_data_ptrs: InputList[Optional[int]] = [
        (
            inputs[i].data_ptr()
            if isinstance(inputs[i], torch.Tensor) and i in self.static_input_idxs
            else None
        )
        for i in range(len(inputs))
    ]

    # When we checkpoint, and free generations, we will be manually freeing the outputs
    # of CUDAGraphNodes. We should not be freeing parameters, not do we need to account for
    # their liveness (they are static), so we need to compute which outputs are aliases of
    # parameters. Some static inputs are saved tensors from the forward that die in the backward.
    # Their locations are static but lifetimes are not. We only include the persistent static
    # data ptrs below because the non persistent data ptrs may be outputs of this record and
    # fresh allocations.

    # precompute expanded dims to avoid computing in the hot path
    self.expanded_dims: List[List[int]] = [
        get_expanded_dims(x)
        if isinstance(x, torch.Tensor) and idx not in self.static_input_idxs
        else []
        for idx, x in enumerate(inputs)
    ]

    # For each node in path, which outputs were observed to be live
    # before invoking graph recording, and after graph recording
    self.recorded_liveness_before_graph: LevelList[OutputList[bool]] = []
    self.recorded_liveness_after_graph: LevelList[OutputList[bool]] = []

    # List of Tuples of (depth, output_index) that index into node at depth
    # number of nodes from root and output_index of outputs. Will index into
    # path_weakrefs.
    self.expected_dead_indices_before_graph: List[PathOutputIndex] = []
    self.expected_dead_indices_after_graph: List[PathOutputIndex] = []

    # all live indices after graph recording
    self.live_indices_after_graph: List[PathOutputIndex] = []

    if self.parent is not None:
        previous_liveness = self.parent.recorded_liveness_after_graph
        curr_liveness = self._get_liveness(self.path_weakrefs)

        different_indices = self._get_different_indices(
            previous_liveness, curr_liveness
        )

        self.recorded_liveness_before_graph = curr_liveness
        self.expected_dead_indices_before_graph = different_indices

    recording_inputs = self._allocate_and_copy_recording_inputs(inputs)
    # recording inputs will copy over memory, so we can free non recording inputs
    inputs.clear()
    del inputs

    # graph used for recording model invocation
    self.graph: Optional[torch.mlu.MLUGraph] = torch.mlu.MLUGraph()

    # we allocate non-static inputs within the same memory pool as the CUDAGraph
    # which we will record the model with. For memory efficiency, it is important
    # to reclaim the input memory when the inputs are no longer live. To accomplish this,
    # we reconstruct tensors at the correct data pointers of our inputs which are
    # non owning and do not prevent deallocation. On subsequent executions, input values
    # will be copied over to these tensors.
    self.reconstructed_inputs: InputList[Union[Tensor, int]] = [
        self._reconstruct_from_tensor_metadata(self._tensor_metadata(x))
        if isinstance(x, torch.Tensor)
        else x
        for x in recording_inputs
    ]

    # DO THE RECORDING!!!
    # We record the CUDA graph in the constructor of CUDAGraphNode, which
    # gives you what the CPU side compute of the function would do.  We
    # don't throw the recording outputs away: their memory is
    # correctly accounted for in the CUDAGraphs caching allocator.  This
    # means on the very FIRST run of the CUDA graph node, we can directly
    # do more recording, because we have a valid caching allocator state.
    # NB: This relies on run() being called immediately after the
    # constructor, otherwise this optimization would not be valid.

    # initialized below in _record

    self.checkpointed_caching_state: Optional[AllocatorState] = None

    # Output Storage Alias information, can be:
    # - A new, unaliased storage, or the output is None
    # - An alias of an output of a prior graph
    # - An alias of an output already created in the reconstructed outputs
    # This is None if the output in question is an int
    self.output_storage_alias: OutputList[Optional[OutputAliasInfo]] = []

    # is the output Storage unaliased in subsequent outputs, of all subsequent paths
    # if it is, we cached the output tensor and adjust storage liveness tracking to also
    # check if the output tensor does not have an additional python reference.
    # If a descendent node discovers it has an alias of a prior output, then the output
    # will no longer be cached in the ancestor.
    # The large majority of tensors are unaliased, and preserving aliased output tensors would add
    # significant additional complexity with marginal gains
    # The cached tensor outputs are added on the first execution, and cleared whenever we need
    # to do subsequent recording
    self.unaliased_in_all_paths: OutputList[bool] = []
    self.cached_tensor_outputs: OutputList[Optional[Tensor]] = []

    # if an output aliases a static, persistent input then the corresponding Tensor will
    # be set here. These are different than cached tensors, because they are tensors that
    # are aliases of parameters that are always live.
    self.static_output_tensors: OutputList[Optional[Tensor]] = []

    # Cleared after recording
    self.recording_outputs: Optional[
        OutputList[Union[torch.Tensor, int]]
    ] = self._record(wrapped_function.model, recording_inputs)
    self.outputs_metadata: OutputList[Union[Dict[str, Any], int, None]] = []

    # As with inputs, we do not want to keep the outputs permanently alive because that would prevent
    # their memory being reclaimed in subsequent cuda graph recordings. We record the tensor metadata
    # needed to reconstruct instead.
    assert self.recording_outputs is not None
    for out in self.recording_outputs:
        if isinstance(out, torch.Tensor):
            self.outputs_metadata.append(
                self._tensor_metadata(out, ignore_storage_offset=False)
            )
        else:
            assert isinstance(out, (int, type(None))), type(out)
            self.outputs_metadata.append(out)

    self.graph.replay()


torch._inductor.cudagraph_trees.CUDAGraphNode.__init__ = CUDAGraphNode__init__


def CUDAGraphNode_record(self, model, inputs):
    "Record the model"

    def static_input_iter():
        for i in self.wrapped_function.static_input_idxs:
            if isinstance(
                inputs[i], torch.Tensor
            ) and not self._is_cuda_graph_recorded_tensor(inputs[i]):
                yield inputs[i]

    # see: output_is_alias_of_persistent_static_inputs above
    static_input_persistent_storage_ptrs: Dict[int, StorageWeakRefWrapper] = {
        inp.untyped_storage().data_ptr(): StorageWeakRefWrapper(inp)
        for inp in itertools.chain(static_input_iter(), self.wrapped_function.constants)
    }

    if config.triton.slow_path_cudagraph_asserts:
        # need to use parent live weakrefs because live_indices isnt set yet
        memory = [] if self.parent is None else list(self.parent.path_live_weakrefs())
        memory += [
            StorageWeakRefWrapper(elem)
            for i, elem in enumerate(inputs)
            if isinstance(elem, torch.Tensor)
            and i not in self.wrapped_function.static_input_idxs
            and elem.untyped_storage().data_ptr() != 0
        ]
        check_memory_pool(self.device, self.cuda_graphs_pool, memory)

    with preserve_rng_state(), torch.mlu.device(self.device), torch.mlu.graph(
        self.graph,
        stream=self.stream,
        pool=self.cuda_graphs_pool,
        capture_error_mode="thread_local",
    ), get_history_recording():
        static_outputs = model(inputs)

    # running model should reclaim memory
    assert len(inputs) == 0

    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    self._add_first_outputs(static_outputs, static_input_persistent_storage_ptrs)

    return static_outputs


torch._inductor.cudagraph_trees.CUDAGraphNode._record = CUDAGraphNode_record


def CUDAGraphNode_add_first_outputs(
    self,
    outputs,
    static_input_persistent_storage_ptrs: Dict[int, StorageWeakRefWrapper],
):
    "Add the outputs from the first invocation of the node and set up metadata"

    # getting liveness before we have added the outputs to path, so the length
    # of the two lists is equal
    prev_liveness = self.recorded_liveness_before_graph
    curr_liveness = self._get_liveness(self.path_weakrefs)

    delta = self._get_different_indices(prev_liveness, curr_liveness)
    self.expected_dead_indices_after_graph = delta

    assert len(self.outputs_weakrefs) == 0
    # index from data pointer to index in outputs
    output_new_storages_index: Dict[StorageDataPtr, int] = {}

    self.unaliased_in_all_paths = [False for _ in range(len(outputs))]
    self.static_output_tensors = [None for _ in range(len(outputs))]

    for i, o in enumerate(outputs):
        if o is None or not isinstance(o, torch.Tensor):
            self.output_storage_alias.append(UnaliasedStorage)
            continue

        torch._check(
            o.is_mlu or o.untyped_storage().data_ptr() == 0,
            lambda: (
                "Expected all mlu outputs in mlu graph recording. Non mlu output "
                f"from {self.stack_traces[i] if self.stack_traces else '(unknown)'}"
            ),
        ),

        ref = static_input_persistent_storage_ptrs.get(
            o.untyped_storage().data_ptr(), None
        )
        # also treat empty storages as static outputs because we do not need to manage their lifetime
        # and they should not participate in checkpointing
        is_empty_storage = o.untyped_storage().data_ptr() == 0
        if (ref and ref() is not None) or is_empty_storage:
            self.output_storage_alias.append(None)
            self.static_output_tensors[i] = o
            continue

        path_ref = self._is_alias_of_live_recorded_tensor(o)
        if path_ref is not None:
            self._mark_prior_graph_output_as_aliased(path_ref)
            self.output_storage_alias.append(AliasesPriorGraphOutput(path_ref))
            continue

        if o.untyped_storage().data_ptr() in output_new_storages_index:
            index = output_new_storages_index[o.untyped_storage().data_ptr()]
            self.unaliased_in_all_paths[index] = False
            self.output_storage_alias.append(AliasesNewOutput(index))
            continue

        output_new_storages_index[o.untyped_storage().data_ptr()] = i
        self.output_storage_alias.append(UnaliasedStorage)
        self.unaliased_in_all_paths[i] = True

    if self.stack_traces is None:
        self.stack_traces = [None for _ in range(len(outputs))]
    else:
        assert len(self.stack_traces) == len(
            outputs
        ), "Wrong number of stack traces passed in"

    assert not self.outputs_weakrefs
    for out, static_output_tensor in zip(outputs, self.static_output_tensors):
        if not isinstance(out, torch.Tensor) or static_output_tensor is not None:
            self.outputs_weakrefs.append(None)
            self.tensor_weakrefs.append(None)
        else:
            self.outputs_weakrefs.append(StorageWeakRefWrapper(out))
            self.tensor_weakrefs.append(TensorWeakRef(out))

    self.recorded_liveness_after_graph = self._get_liveness(self.path_weakrefs)
    self.checkpointed_caching_state = torch_mlu._MLUC._mlu_getCheckpointState(
        self.device, self.cuda_graphs_pool
    )

    # now, get liveness with outputs added
    for depth in range(len(self.path_weakrefs)):
        for output_index in range(len(self.path_weakrefs[depth])):
            if is_live(self.path_weakrefs[depth][output_index]):
                self.live_indices_after_graph.append((depth, output_index))

    self.debug_check_invariants_after_invocation()
    if config.triton.slow_path_cudagraph_asserts:
        check_memory_pool(
            self.device, self.cuda_graphs_pool, list(self.path_live_weakrefs())
        )


torch._inductor.cudagraph_trees.CUDAGraphNode._add_first_outputs = (
    CUDAGraphNode_add_first_outputs
)


def CUDAGraphNode_initialize_cached_tensors(self):
    # we should not be clearing output_weakrefs, and they should be set in the first
    # record run
    assert len(self.outputs_weakrefs) == len(self.outputs_metadata)

    for i, (storage_info, metadata, make_cached) in enumerate(
        zip(
            self.output_storage_alias,
            self.outputs_metadata,
            self.unaliased_in_all_paths,
        )
    ):
        if not make_cached:
            self.cached_tensor_outputs.append(None)
            continue

        assert storage_info is UnaliasedStorage
        assert isinstance(metadata, dict)
        s = self.create_storage(metadata)
        out = self._reconstruct_from_tensor_metadata(metadata, storage=s)

        # XXX: let autograd know that there will be an additional reference to the tensor
        # that can be ignored when deciding whether to do gradient buffer inplacing.
        # Otherwise, inplacing could differ between tracing and subsequent execution.
        # For some models we tested this led to inputs no longer being in cudagraph pools,
        # leading to spurious re-recordings.
        # It also tells AMP cache that even though the tensor impls cannot be cached
        # in dtype conversions.

        torch_mlu._MLUC._add_cached_tensor(out)

        self_ref = weakref.ref(self)

        # one reference in our array, and calling sys.getrefcount bumps the refcount by one
        def check_refcount(i):
            self_loc = self_ref()
            if self_loc is None:
                return False
            return self_loc.get_output_refcount(i) == 2

        check = functools.partial(check_refcount, i=i)

        self.outputs_weakrefs[i] = StorageWeakRefWrapper(out, extra_ref_check=check)
        self.cached_tensor_outputs.append(out)


torch._inductor.cudagraph_trees.CUDAGraphNode._initialize_cached_tensors = (
    CUDAGraphNode_initialize_cached_tensors
)


def CUDAGraphNode_remove_node_cached_tensors(self):
    for t in self.cached_tensor_outputs:
        if t is not None:
            torch_mlu._MLUC._remove_cached_tensor(t)
    self.cached_tensor_outputs.clear()

    for i, unaliased in enumerate(self.unaliased_in_all_paths):
        if unaliased:
            n = self.outputs_weakrefs[i]
            assert n is not None
            n.remove_extra_reference()


torch._inductor.cudagraph_trees.CUDAGraphNode.remove_node_cached_tensors = (
    CUDAGraphNode_remove_node_cached_tensors
)


def _reconstruct_from_tensor_metadata(
    self, metadata: Dict[str, Any], storage=None
) -> Tensor:
    s = self.create_storage(metadata) if storage is None else storage
    return torch_mlu._MLUC._construct_MLU_Tensor_From_Storage_And_Metadata(metadata, s)


torch._inductor.cudagraph_trees.CUDAGraphNode._reconstruct_from_tensor_metadata = (
    _reconstruct_from_tensor_metadata
)


def _allocate_and_copy_recording_inputs(self, inputs) -> List[Union[torch.Tensor, int]]:
    """
    Allocate inputs for non static, non cudagraph managraphed managed tensors in the memory pool
    and copy over the tensor values.
    """

    torch.mlu.synchronize()
    self.stream.wait_stream(torch.mlu.current_stream())
    recording_inputs: List[Union[Tensor, int]] = []

    with warnings.catch_warnings(record=True), torch.mlu.device(
        self.device
    ), _use_cuda_memory_pool_manager(
        self.device,
        mem_pool=self.cuda_graphs_pool,
        stream=self.stream,
    ):
        for i, inp in enumerate(inputs):
            if not isinstance(inp, torch.Tensor):
                assert isinstance(inp, int)
                recording_inputs.append(inp)
            elif i not in self.static_input_idxs:
                # static_input does an allocation!
                recording_inputs.append(static_input(inp))
                # copy over and clear non recording input
                self._copy_input(i, recording_inputs[-1], inp)
                inputs[i] = None
                del inp
            else:
                recording_inputs.append(inp)

    return recording_inputs


torch._inductor.cudagraph_trees.CUDAGraphNode._allocate_and_copy_recording_inputs = (
    _allocate_and_copy_recording_inputs
)


def get_cudagraph_segments(pool_id):
    segments = torch.mlu.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


torch._inductor.cudagraph_trees.get_cudagraph_segments = get_cudagraph_segments


def check_memory_pool(device, pool_id, live_storages_ptrs: List[StorageWeakRefWrapper]):
    assert all(
        isinstance(elem, StorageWeakRefWrapper) for elem in live_storages_ptrs
    )  # noqa: C419
    unique_storages = {stor.data_ptr() for stor in live_storages_ptrs if stor()}

    # check if there is a divergence first, then do the expensive snapshot call after
    # we know it will error
    if torch_mlu._MLUC._mlu_checkPoolLiveAllocations(device, pool_id, unique_storages):
        return

    # at this point we are past the fast-path. we have seen rare cases where a dead tensor is dead,
    # but hasn't been gc'd yet, and gives false positive for allocated_not_in_live_storages
    gc.collect()

    segments = get_cudagraph_segments(pool_id)

    allocated_not_in_live_storages = {}

    for segment in segments:
        addr = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                if addr not in unique_storages:
                    allocated_not_in_live_storages[addr] = block
                else:
                    unique_storages.remove(addr)

            addr += block["size"]

    torch._check(
        len(unique_storages) == 0,
        lambda: f"These storage data ptrs are not allocated in pool {pool_id} but should be {unique_storages}",
    )

    if allocated_not_in_live_storages != 0:
        formatted = []
        for dp, block in allocated_not_in_live_storages.items():
            trace = format_tb(block.get("frames", []))
            formatted.append(f"Data Pointer: {dp}, history: \n{trace}")
        formatted_s = "\n".join(formatted)
        msg = (
            f"These live storage data ptrs are in the cudagraph pool but not "
            f"accounted for as an output of cudagraph trees: \n\n{formatted_s}"
        )
        raise RuntimeError(msg)


torch._inductor.cudagraph_trees.check_memory_pool = check_memory_pool


def CUDAGraphTreeManager__init__(self, device_index: int):
    # roots are functions which have no dependencies on an other node. I.e.,
    # when they are first invoked, none of their inputs are outputs are outputs
    # of another node, nor are there any live outputs of another node whose
    # liveness would create a dependency.
    self.roots: Dict[FunctionID, List[CUDAGraphNode]] = defaultdict(list)

    # mapping from function id to wrapped function
    self.ids_to_funcs: Dict[FunctionID, WrappedFunction] = {}

    self.ids_to_stack_traces: Dict[FunctionID, StackTraces] = {}

    self.warmed_up_functions: Set[FunctionID] = set()
    # if we fail to increment generation, and are stuck warming up,
    # only warn on each function once
    self.warned_functions: Set[FunctionID] = set()
    torch_mlu._MLUC._set_cached_tensors_enabled(True)

    # NB: cuda caching allocator will remember the stream a segment is allocated to
    # and only allocate that segment to the same stream. we need to use a single stream
    # for all allocations to the memory pool, otherwise the allocations to separate streams
    # will not be reused; separate recordings would have use the same memory pool, but not
    # the same memory.

    with torch.mlu.device(device_index):
        torch.mlu.synchronize()
        self.stream = torch.mlu.Stream()
        self.stream.wait_stream(torch.mlu.current_stream())

        # Keeps Memory Pool Alive
        self.graph: Optional[torch.mlu.MLUGraph] = torch.mlu.MLUGraph()
        self.cuda_graphs_thread_pool = torch.mlu.graph_pool_handle()

        with warnings.catch_warnings(record=True), torch.mlu.graph(
            self.graph,
            pool=self.cuda_graphs_thread_pool,
            stream=self.stream,
            capture_error_mode="thread_local",
        ):
            pass

    self.graph_counter = itertools.count(0)
    self.func_counter = itertools.count(0)

    # whether we the current node is in a state of warmup, recording, execution. If
    # there is no current node the state will be ExecutionState.None.
    self.path_state = ExecutionState.NONE
    self.device_index = device_index

    # the most recently invoked cudagraph wrapping of a function. Will be None
    # when there is no output from a previous recording or execution whose memory
    # we need to respect in the cuda caching allocation. If you incremented generation,
    # this will also be none, as ignore those allocations.
    self.current_node: Optional[CUDAGraphNode] = None

    # current generation of cudagraph invocations. when torch.compile is run
    # we increment the current generation. are willing to ignore live outputs
    # of a previous generation in checking liveness.
    self.current_gen: int = -1

    # number of instances we are in execution and failed to match to an
    # existing child
    self.debug_fail_counter = 0
    # number of instances we had to checkpoint the function
    self.debug_checkpointing_counter = 0

    self.id_to_mode: Dict[FunctionID, CompilationMode] = {}

    # Note: [Backward Generation Handling]
    # We generally perform a sequence of forward executions followed by backward executions.
    # If multiple torch.compile wrapped forwards are executed with their backwards pending,
    # we should not disregard the outputs from a prior torch.compile since the entire training
    # loop hasn't completed.  Occasionally, a backward pass corresponding to a forward pass may
    # not be executed, so we cannot wait for all pending forward pass backward completions, so
    # we cannot wait for all backwards to have been invoked. Instead we wait for a single backward
    # invocation. Triggering a backward pass typically doesn't lead to another torch.compile
    # invocation, making it less likely for the generation to increase between multiple
    # backward calls. The following use case is covered by this approach:
    # mod1 = torch.compile(...)
    # mod2 = torch.compile(...)
    # mod2(mod1(x)).sum().backward()

    self.running_forwards_with_pending_backwards = False


torch._inductor.cudagraph_trees.CUDAGraphTreeManager.__init__ = (
    CUDAGraphTreeManager__init__
)


def CUDAGraphTreeManager_record_function(
    self, new_inputs, function_id
) -> List[Optional[Tensor]]:
    graph_id = self.new_graph_id()
    log.debug(
        "Recording function %d of graph recording id %d",
        function_id.id,
        graph_id.id,
    )
    torch.mlu.synchronize()
    node = CUDAGraphNode(
        self.ids_to_funcs[function_id],
        graph_id,
        self.current_node,
        new_inputs,
        self.cuda_graphs_thread_pool,
        self.device_index,
        self.ids_to_stack_traces[function_id],
        self.stream,
    )
    if self.current_node is None:
        self.roots[function_id].append(node)
    else:
        self.current_node.add_child(function_id, node)
    self.current_node = node
    self.path_state = ExecutionState.RECORDING
    self.update_generation()
    torch.mlu.synchronize()
    return node.run_first_inputs(new_inputs)


torch._inductor.cudagraph_trees.CUDAGraphTreeManager.record_function = (
    CUDAGraphTreeManager_record_function
)


def CUDAGraphTreeManager_add_function(
    self,
    model,
    inputs,
    static_input_idxs,
    stack_traces,
    mode,
    constants,
) -> Tuple[Callable[..., Any], List[Optional[Tensor]]]:
    id = self.new_func_id()
    self.ids_to_stack_traces[id] = stack_traces
    self.ids_to_funcs[id] = WrappedFunction(
        model,
        static_input_idxs,
        id,
        tuple(t for t in constants if isinstance(t, torch.Tensor) and t.is_mlu),
    )
    self.id_to_mode[id] = mode
    fn = functools.partial(self.run, function_id=id)

    # container needs to set clean up when fn dies
    get_container(self.device_index).add_strong_reference(fn)
    return fn, fn(inputs)


torch._inductor.cudagraph_trees.CUDAGraphTreeManager.add_function = (
    CUDAGraphTreeManager_add_function
)


def CUDAGraphTreeManager_apply_checkpoint_execution_state_in_allocator(self):
    """
    Checkpoint the current execution state in the caching allocator so that
    additional cudagraph recordings can be made respecting existent live storages.
    """
    self.debug_checkpointing_counter += 1
    log.debug(
        "Checkpointing cuda caching allocator state. Number of checkpoints %d",
        self.debug_checkpointing_counter,
    )

    state = self.current_node.checkpointed_caching_state
    device = self.current_node.device
    assert state is not None and device is not None

    # currently we deallocate on instead of allowing stale recordings
    stale_storages: List[int] = []

    # remove cached tensors, otherwise they would prevent memory from being
    # reclaimed in subsequent recordings
    self.current_node.remove_path_cached_tensors()
    live_storages_wrappers = list(self.current_node.path_live_weakrefs())

    live_storages_weak_refs = [t() for t in live_storages_wrappers]
    ptrs_to_deallocate = self.current_node.data_ptrs_dead_since_invocation()
    torch_mlu._MLUC._mlu_setCheckpointPoolState(
        device, state, stale_storages, live_storages_weak_refs
    )

    # NB: deduplicate aliased outputs
    for ptr in set(ptrs_to_deallocate):
        torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete(ptr)

    # Now the live blocks should be exactly equal to the live storages in private pool
    if config.triton.slow_path_cudagraph_asserts:
        check_memory_pool(
            self.device_index, self.cuda_graphs_thread_pool, live_storages_wrappers
        )
        for wrapper in live_storages_wrappers:
            assert wrapper()
            assert torch_mlu._MLUC._has_Standard_Deleter(wrapper())
            assert wrapper.data_ptr() not in ptrs_to_deallocate


torch._inductor.cudagraph_trees.CUDAGraphTreeManager.apply_checkpoint_execution_state_in_allocator = (
    CUDAGraphTreeManager_apply_checkpoint_execution_state_in_allocator
)


def CUDAGraphTreeManager_dealloc_current_path_weakrefs(self):
    # TODO: we could also allow the these weak refs to continue to be allocated,
    # but that adds some complications.
    for node in self.current_node._path_from_root:
        assert len(node.tensor_weakrefs) == len(node.stack_traces)
        for t, stack_trace in zip(node.tensor_weakrefs, node.stack_traces):
            ten = None if t is None else t()
            if ten is None:
                continue

            stack_trace = (
                stack_trace.strip() if stack_trace else "[Could not find stack trace]"
            )
            msg = (
                "Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. "
                f"Stack trace: {stack_trace}. "
                "To prevent overwriting, clone the tensor outside of torch.compile() "
                "or call torch.compiler.cudagraph_mark_step_begin() before each model invocation."
            )
            torch_mlu._MLUC._set_storage_access_error_msg(ten, msg)

    deleted = set()
    for storage_ref in self.current_node.path_live_weakrefs():
        if storage_ref() and storage_ref.data_ptr() not in deleted:
            deleted.add(storage_ref.data_ptr())
            torch_mlu._MLUC._free_And_Remove_DeleterFn(storage_ref())


torch._inductor.cudagraph_trees.CUDAGraphTreeManager.dealloc_current_path_weakrefs = (
    CUDAGraphTreeManager_dealloc_current_path_weakrefs
)
