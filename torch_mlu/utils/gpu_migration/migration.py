from functools import wraps
import functools
from typing import Callable, Dict, List, Tuple, Union
import warnings
import sys
import inspect
import types
import os

import torch
import torch.autograd.profiler as prof
import torch.distributed.distributed_c10d as c10d
import torch.distributed.fsdp.sharded_grad_scaler

import torch_mlu
import torch_mlu.distributed.fsdp.sharded_grad_scaler

# warning once
warnings.filterwarnings(action="once")

# class can accept device arg
class_list = [
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.LazyConv1d,
    torch.nn.LazyConv2d,
    torch.nn.LazyConv3d,
    torch.nn.LazyConvTranspose1d,
    torch.nn.LazyConvTranspose2d,
    torch.nn.LazyConvTranspose3d,
    torch.nn.MultiheadAttention,
    torch.nn.PReLU,
    torch.nn.AdaptiveLogSoftmaxWithLoss,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.LazyBatchNorm1d,
    torch.nn.LazyBatchNorm2d,
    torch.nn.LazyBatchNorm3d,
    torch.nn.GroupNorm,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LazyInstanceNorm1d,
    torch.nn.LazyInstanceNorm2d,
    torch.nn.LazyInstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.RNNBase,
    torch.nn.RNN,
    torch.nn.LSTM,
    torch.nn.GRU,
    torch.nn.RNNCell,
    torch.nn.LSTMCell,
    torch.nn.GRUCell,
    torch.nn.Transformer,
    torch.nn.TransformerEncoderLayer,
    torch.nn.TransformerDecoderLayer,
    torch.nn.Linear,
    torch.nn.Bilinear,
    torch.nn.LazyLinear,
    torch.nn.Embedding,
    torch.nn.EmbeddingBag,
    # DDP
    torch.nn.parallel.DistributedDataParallel,
    # amp
    torch.amp.autocast,
    torch.utils._device.DeviceContext,
]


# torch.* that needs to replace device, selected from torch/_torch_docs.py
torch_fn_list = [
    "set_default_device",
    "as_tensor",
    "asarray",
    "autocast",
    "eye",
    "linspace",
    "logspace",
    "load",
    "ones",
    "ones_like",
    "rand",
    "rand_like",
    "randint",
    "randint_like",
    "randn",
    "randn_like",
    "randperm",
    "tensor",
    "range",
    "arange",
    "sparse_compressed_tensor",
    "sparse_csr_tensor",
    "sparse_csc_tensor",
    "sparse_bsr_tensor",
    "sparse_bsc_tensor",
    "sparse_coo_tensor",
    "tril_indices",
    "triu_indices",
    "zeros",
    "zeros_like",
    "empty",
    "empty_like",
    "empty_strided",
    "full",
    "full_like",
    "hann_window",
    "hamming_window",
    "bartlett_window",
    "blackman_window",
    "kaiser_window",
    # the following torch functions are not listed in _torch_docs.py,
    # but can still use torch.* (selected from native_functions.yaml).
    "_cudnn_init_dropout_state",
    "_empty_affine_quantized",
    "_empty_per_channel_affine_quantized",
    "empty_quantized",
    "from_file",
    "_pin_memory",
    "scalar_tensor",
    "_efficientzerotensor",
    "_sparse_compressed_tensor_unsafe",
    "_sparse_csr_tensor_unsafe",
    "_sparse_csc_tensor_unsafe",
    "_sparse_bsr_tensor_unsafe",
    "_sparse_bsc_tensor_unsafe",
    "_sparse_coo_tensor_unsafe",
    "normal",
    "_nested_tensor_from_tensor_list",
    # they are not available in the torch module
    # 'fft_fftfreq',
    # 'fft_rfftfreq',
    # '_sparse_coo_tensor_with_dims'
    # '_sparse_coo_tensor_with_dims_and_tensors',
]

torch_cuda_fn_list = [
    "set_device",
    "get_device_name",
    "get_device_capability",
    "get_device_properties",
    "can_device_access_peer",
    "synchronize",
    "current_stream",
    "default_stream",
    # skip nvml fn
    # '_get_nvml_device_index',
    # '_get_pynvml_handler',
    # 'memory_usage',
    # 'utilization',
    # 'temperature',
    # 'power_draw',
    # 'clock_rate',
    "list_gpu_processes",
    "mem_get_info",
    "memory_stats",
    "memory_summary",
    "memory_allocated",
    "max_memory_allocated",
    "reset_max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "set_per_process_memory_fraction",
    "memory_cached",
    "max_memory_cached",
    "reset_max_memory_cached",
    "reset_peak_memory_stats",
]

torch_cuda_memory_fn_list = [
    "mem_get_info",
    "memory_stats",
    "memory_summary",
    "memory_allocated",
    "max_memory_allocated",
    "reset_max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "set_per_process_memory_fraction",
    "memory_cached",
    "max_memory_cached",
    "reset_max_memory_cached",
    "reset_peak_memory_stats",
]

# torch.Tensor.* that needs to replace device, selected from torch/_tensor_docs.py
tensor_fn_list = [
    "new_tensor",
    "new_full",
    "new_empty",
    "new_empty_strided",
    "new_ones",
    "new_zeros",
    "to",
    "is_pinned",
    "pin_memory",
    "mlu",
    "new",
]

distributed_fn_list = [
    "broadcast_object_list",
    "init_device_mesh",
]

module_fn_list = [
    "to",
    "to_empty",
]


default_cuda_args_list = [
    "torch.random.fork_rng",
]


# this list is for functions whose modules are not in torch, torch.distributed and torch.module
# TODO: Merge all fn lists into one list
other_fn_list = [
    {
        "name": "torch.random.fork_rng",
        "mod": torch.random,
        "fn_list": [
            "fork_rng",
        ],
    },
    {
        "name": "torch.utils.cpp_extension.include_paths",
        "mod": torch.utils.cpp_extension,
        "fn_list": [
            "include_paths",
        ],
    },
    {
        "name": "torch.distributed.Backend.register_backend",
        "mod": torch.distributed.Backend,
        "fn_list": [
            "register_backend",
        ],
    },
    {
        "name": "torch.distributed.tensor.parallel.parallelize_module",
        "mod": torch.distributed.tensor.parallel,
        "fn_list": [
            "parallelize_module",
        ],
    },
    {
        "name": "torch.testing.make_tensor",
        "mod": torch.testing,
        "fn_list": [
            "make_tensor",
        ],
    },
    {
        "name": "torch.amp.autocast_mode.is_autocast_available",
        "mod": torch.amp.autocast_mode,
        "fn_list": [
            "is_autocast_available",
        ],
    },
    {
        "name": "torch.amp.custom_bwd",
        "mod": torch.amp,
        "fn_list": [
            "custom_bwd",
        ],
    },
    {
        "name": "torch.amp.custom_fwd",
        "mod": torch.amp,
        "fn_list": [
            "custom_fwd",
        ],
    },
]


# add new tuples here to adapt new substitution
substitution_list = [
    ("nccl", "cncl"),
    ("nvtx", "cnpx"),
    ("CUDAGraph", "MLUGraph"),
    ("is_available", "is_cncl_available"),
    ("CUDAPluggableAllocator", "MLUPluggableAllocator"),
]


# handle profiler, set use_cuda=False
def prepare_trace(self):
    self.profiler = prof.profile(
        use_cuda=False,
        use_cpu=(torch.profiler.ProfilerActivity.CPU in self.activities),
        use_mtia=(torch.profiler.ProfilerActivity.MTIA in self.activities),
        use_device=None,
        record_shapes=self.record_shapes,
        with_flops=self.with_flops,
        profile_memory=self.profile_memory,
        with_stack=self.with_stack,
        with_modules=self.with_modules,
        use_kineto=True,
        experimental_config=self.experimental_config,
        use_mlu=(torch.profiler.ProfilerActivity.MLU in self.activities),
    )
    self.profiler._prepare_trace()


class Generator(torch.Generator):
    def __new__(cls, device: Union[torch.device, str, None] = None):
        return super().__new__(cls, device)


def replace_device_args(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                args[i] = replace_cuda_with_mlu(arg)

        if kwargs:
            # device=* , device_type=*, output_device=*
            # type can be str, torch.device or int
            for key in ["device", "device_type", "output_device", "map_location"]:
                device = kwargs.get(key, None)
                if device:
                    kwargs[key] = replace_cuda_with_mlu(device)

            # list of int or torch.device
            device_ids = kwargs.get("device_ids", None)
            if isinstance(device_ids, list):
                for i, arg in enumerate(device_ids):
                    device_ids[i] = replace_cuda_with_mlu(arg)
                kwargs["device_ids"] = device_ids
        return fn(*args, **kwargs)

    return wrapper_fn


def replace_cuda_with_mlu(arg):
    # 'cuda*' -> 'mlu*'
    if isinstance(arg, str) and "cuda" in arg:
        arg = arg.replace("cuda", "mlu")
    # torch.device('cuda*') -> torch.device('mlu*')
    if isinstance(arg, torch.device) and "cuda" in arg.type:
        device = f"mlu:{arg.index}" if arg.index is not None else "mlu"
        arg = torch.device(device)
    # int ids are no need to handle, because relative
    # modifications have already patched to Pytorch
    return arg


def replace_device(module, fn_list):
    for fn_name in fn_list:
        fn = getattr(module, fn_name, None)
        if fn:
            setattr(module, fn_name, replace_device_args(fn))


# init_process_group
# 1. 'nccl' -> 'cncl'
# 2. backend='nccl' -> backend='cncl'
def replace_nccl_with_cncl(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, str) and "nccl" in arg:
                    args[i] = arg.replace("nccl", "cncl")
        if kwargs:
            backend = kwargs.get("backend", None)
            if isinstance(backend, str) and "nccl" in backend:
                kwargs["backend"] = "cncl"
        return fn(*args, **kwargs)

    return wrapper_fn


# profiler
# sort_by='*cuda*' -> sort_by='*mlu*'
# use_cuda=True  ->  use_mlu=True
def replace_profiler_args(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                args[i] = replace_cuda_with_mlu(arg)

        if kwargs:
            device = kwargs.get("sort_by", None)
            if device:
                kwargs["sort_by"] = replace_cuda_with_mlu(device)
            use_cuda = kwargs.get("use_cuda", None)
            if use_cuda:
                kwargs["use_cuda"] = False
                kwargs["use_mlu"] = True
        return fn(*args, **kwargs)

    return wrapper_fn


def replace_profiler_legacy_args(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if kwargs:
            use_cuda = kwargs.get("use_cuda", None)
            if use_cuda:
                raise NotImplementedError(
                    "MLU not support torch.autograd.profiler_legacy.profile(), and this class will be deprecated. Use torch.profiler instead"
                )
        return fn(*args, **kwargs)

    return wrapper_fn


def replace_profiler(module, fn_name):
    fn = getattr(module, fn_name, None)
    if fn:
        setattr(module, fn_name, replace_profiler_args(fn))


def replace_profiler_legacy(module, fn_name):
    fn = getattr(module, fn_name, None)
    if fn:
        setattr(module, fn_name, replace_profiler_legacy_args(fn))


# dataloader
# pin_memory_device=''->pin_memory_device='mlu'
# pin_memory_device='cuda'->pin_memory_device='mlu'
def replace_dataloader(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if kwargs:
            pin_memory = kwargs.get("pin_memory", False)
            pin_memory_device = kwargs.get("pin_memory_device", None)
            if pin_memory and not pin_memory_device:
                kwargs["pin_memory_device"] = "mlu"
            if (
                pin_memory
                and isinstance(pin_memory_device, str)
                and "cuda" in pin_memory_device
            ):
                kwargs["pin_memory_device"] = pin_memory_device.replace("cuda", "mlu")
        return fn(*args, **kwargs)

    return wrapper_fn


def script(
    obj,
    optimize=None,
    _frames_up=0,
    _rcb=None,
    example_inputs: Union[List[Tuple], Dict[Callable, List[Tuple]], None] = None,
):
    warnings.warn(
        f"The model-transfer tool does not support torch.jit.script and use eager mode instead."
    )
    return obj


def _new_privateuse1_deserialize(obj, location):
    if location.startswith("cuda"):
        location = location.replace("cuda", "mlu")
    return torch.serialization._privateuse1_deserialize(obj, location)


torch.serialization.register_package(
    11, torch.serialization._privateuse1_tag, _new_privateuse1_deserialize
)

old_device_constructors_ = torch.utils._device._device_constructors()


@functools.lru_cache(1)
def original_device_constructors():
    global old_device_constructors_
    return old_device_constructors_


# This function is used to check whether an object needs to be patched (called by `update_dict``)
def needs_skip(obj, module):
    # filter out functions which are not defined in current module
    if isinstance(obj, (types.FunctionType, type)) and (
        obj.__module__.startswith(module.__name__) or obj.__module__.startswith("torch")
    ):
        return False
    # filter out modules like os, List, Union
    elif isinstance(obj, types.ModuleType) and (
        ".".join(obj.__name__.split(".")[:-1]).startswith(module.__name__)
        or ".".join(obj.__name__.split(".")[:-1]).startswith("torch")
    ):
        return False
    return True


# Here we only need to ensure that the two input obj are both a class/func rether than
# both of the two inputs have exactly the same type
def rough_type_comparison(obj1, obj2):
    if isinstance(obj1, types.ModuleType) and isinstance(obj2, types.ModuleType):
        return True
    elif inspect.isclass(obj2) and inspect.isclass(obj2):
        return True
    elif (inspect.isfunction(obj1) or isinstance(obj1, types.BuiltinFunctionType)) and (
        inspect.isfunction(obj2) or isinstance(obj2, types.BuiltinFunctionType)
    ):
        return True
    elif inspect.ismethod(obj1) and inspect.ismethod(obj2):
        return True
    return False


# Check whether all of the modules in __all__ have been wrapped/ patched. If not, a warning will be provided
def check_patch_succeeded(source_module, dest_module):
    if hasattr(source_module, "__all__"):
        for attr in source_module.__all__:
            # process substitution list
            attr_dst = attr
            for dst_sub, src_sub in substitution_list:
                if attr == src_sub:
                    attr_dst = dst_sub
            try:
                if not isinstance(dest_module.__dict__[attr], types.ModuleType):
                    if dest_module.__dict__[attr_dst] != source_module.__dict__[attr]:
                        warnings.warn(
                            f"{source_module.__dict__[attr].__name__} is supported in {source_module.__name__} and not patched"
                            f" in {dest_module.__name__}. Please check whether {source_module.__dict__[attr].__name__} needs to be migrated."
                        )
            except:
                pass


# for unsupported funcs/classes/modules, we add wrappers.
def warning_wrapper(obj, obj_name):
    warning_message = "gpu_migration: " + obj_name + " is not yet suppoorted on MLU"

    def decorator(obj):
        # decorate func
        if isinstance(obj, types.FunctionType):

            @wraps(obj)
            def wrapped_function(*args, **kwargs):
                warnings.warn(warning_message, UserWarning)
                return obj(*args, **kwargs)

            return wrapped_function

        # decorate classes
        elif isinstance(obj, type):

            class WrappedClass(obj):
                def __init__(self, *args, **kwargs):
                    warnings.warn(warning_message, UserWarning)
                    super().__init__(*args, **kwargs)

                # for callable classes
                def __call__(self, *args, **kwargs):
                    warnings.warn(warning_message, UserWarning)
                    return super().__call__(*args, **kwargs)

            return WrappedClass

        # decorate callable obj (e.g. an instance of a callable class)
        elif callable(obj):

            class WrappedCallable:
                def __init__(self, callable_obj):
                    self._callable_obj = callable_obj

                def __call__(self, *args, **kwargs):
                    warnings.warn(warning_message, UserWarning)
                    return self._callable_obj(*args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._callable_obj, name)

            return WrappedCallable(obj)

        # return original module
        else:
            return obj

    return decorator(obj)


# This func is used to apply monkey patches and update sys.modules and module.__dict__
# All attrs that are in dest_module.__dict__ but not in source_module.__dict__ will be
# wrapped(warning added) and added to source_module.__dict__. In addition, sys.modules[dest_module]
# will be set to source_module.
def update_dict(source_module, dest_module):
    source_dict = source_module.__dict__
    dest_dict = dest_module.__dict__
    # torch.cuda
    updates = {}
    for dest_key in dest_dict.copy():
        patched = False
        # filter out all attrs that not begins with "__" and not defined in current module
        if dest_key.startswith("__") or needs_skip(dest_dict[dest_key], dest_module):
            continue
        # torch_mlu.mlu
        for source_key in source_dict:
            if source_key == dest_key or (dest_key, source_key) in substitution_list:
                # Report a warning and skip the current substitution when two eponymous have different types
                if not rough_type_comparison(
                    source_dict[source_key], dest_dict[dest_key]
                ):
                    warnings.warn(
                        f"Warning: The object {source_dict[source_key]} in source_module {source_module.__name__} is a {type(source_dict[source_key])}, "
                        f"but in dest_module {dest_module.__name__}, it is a {type(dest_dict[dest_key])}. Please align the types."
                        f" This substitution will be skipped"
                    )
                    # Add a warning msg and process the next obj
                    break
                else:
                    # process modules
                    if isinstance(source_dict[source_key], types.ModuleType):
                        update_dict(source_dict[source_key], dest_dict[dest_key])
                        updates[dest_key] = source_dict[source_key]
                        patched = True
                        break
                    # process classes and funcs
                    else:
                        # Do nothing when dest_key equals source_key
                        updates[dest_key] = source_dict[source_key]
                        patched = True
                        break

        if not patched and os.getenv("TORCH_MLU_MIGRATION_WITH_WARNING", "1") == "1":
            updates[dest_key] = warning_wrapper(dest_dict[dest_key], dest_key)
    # update __dict__
    source_dict.update(updates)

    # get relative module name and parent module
    parent_module_name = ".".join(dest_module.__name__.split(".")[:-1])

    # for 'root' modules, will not be run
    if not parent_module_name:
        sys.modules[dest_module.__name__] = source_module
        return

    parent_module = sys.modules[parent_module_name]
    dest_module_name = dest_module.__name__.removeprefix(parent_module.__name__ + ".")

    # monkey patches for modules
    setattr(parent_module, dest_module_name.split(".")[-1], source_module)
    parent_module.__dict__[dest_module_name.split(".")[-1]] = source_module
    sys.modules[dest_module.__name__] = source_module


def replace_default_cuda_args_with_mlu(fn):
    @wraps(fn)
    def warp_fn(*args, **kwargs):
        sig = inspect.signature(fn)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        device = bound_args.arguments.get("device", None)
        if device == "cuda":
            bound_args.arguments["device"] = "mlu"

        device_type = bound_args.arguments.get("device_type", None)
        if device_type == "cuda":
            bound_args.arguments["device_type"] = "mlu"

        return fn(*bound_args.args, **bound_args.kwargs)

    return warp_fn


def replace_default_cuda_args(func_name):
    components = func_name.split(".")
    module = torch
    fn_name = components[-1]
    for comp in components[1:-1]:
        module = getattr(module, comp, None)
    if module is None:
        raise AttributeError(f"module '{func_name}' does not exist")
    fn = getattr(module, fn_name, None)
    if fn is not None:
        setattr(module, fn_name, replace_default_cuda_args_with_mlu(fn))


def apply_monkey_patches():
    # replace order MATTERS, this functions need to be replaced before torch_fn_list
    torch.utils._device._device_constructors = original_device_constructors
    # torch.*
    replace_device(torch, torch_fn_list)

    torch._ops.OpOverload.__call__ = replace_device_args(torch._ops.OpOverload.__call__)
    torch._ops.OpOverloadPacket.__call__ = replace_device_args(
        torch._ops.OpOverloadPacket.__call__
    )

    torch.get_autocast_gpu_dtype = torch.mlu.get_autocast_dtype
    torch.set_autocast_gpu_dtype = torch.mlu.set_autocast_dtype
    torch.is_autocast_enabled = torch.mlu.is_autocast_enabled
    torch._C._cuda_getCurrentRawStream = torch_mlu._MLUC._mlu_getCurrentRawStream

    # torch.cuda.*
    update_dict(torch.mlu, torch.cuda)
    check_patch_succeeded(torch_mlu.mlu, torch.cuda)
    # TODO(PYTORCH-12705) : need to conside other api like: torch.cuda.memory.xxx
    replace_device(torch.cuda.memory, torch_cuda_memory_fn_list)
    replace_device(torch.cuda, torch_cuda_fn_list)

    # torch.Tensor.*
    replace_device(torch.Tensor, tensor_fn_list)
    torch.Tensor.cuda = torch.Tensor.mlu
    torch.Tensor.is_cuda = torch.Tensor.is_mlu

    # some functions like torch.Tensor.to is first recorded in _allowed_methods
    # and then wrapped here, so the method in _allowed_methods needs to be updated.
    for i, method in enumerate(
        torch.nn.parameter.UninitializedTensorMixin._allowed_methods
    ):
        if method.__name__ in tensor_fn_list and method.__name__ != "mlu":
            torch.nn.parameter.UninitializedTensorMixin._allowed_methods[i] = getattr(
                torch.Tensor, method.__name__
            )

    # torch.nn.Module.*
    replace_device(torch.nn.Module, module_fn_list)
    torch.nn.Module.cuda = torch.nn.Module.mlu

    for mod in class_list:
        replace_device(mod, ["__init__"])

    # torch.Generator
    # can't set attributes of extension type 'torch._C.Generator',
    # so we use a subclass of Generator
    torch.Generator = Generator
    replace_device(torch.Generator, ["__new__"])

    # torch.distributed.init_process_group
    c10d.init_process_group = replace_nccl_with_cncl(c10d.init_process_group)
    torch.distributed.init_process_group = replace_nccl_with_cncl(
        torch.distributed.init_process_group
    )
    c10d.new_group = replace_nccl_with_cncl(c10d.new_group)
    torch.distributed.new_group = replace_nccl_with_cncl(torch.distributed.new_group)
    c10d.is_nccl_available = torch.distributed.is_cncl_available
    torch.distributed.is_nccl_available = torch.distributed.is_cncl_available
    torch.distributed.ProcessGroup._get_backend = replace_device_args(
        torch.distributed.ProcessGroup._get_backend
    )
    c10d.ProcessGroupNCCL = torch_mlu._MLUC.ProcessGroupCNCL
    torch.distributed.ProcessGroupNCCL = torch_mlu._MLUC.ProcessGroupCNCL
    # If the backend is not provided, then both a gloo
    # and nccl backend will be created by init_process_group,
    # we create gloo and cncl instead of gloo and nccl.
    c10d.Backend.default_device_backend_map.pop("cuda", "nccl")

    # fsdp
    torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler = (
        torch_mlu.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler
    )

    # torch.utils.data.DataLoader
    torch.utils.data.DataLoader.__init__ = replace_dataloader(
        torch.utils.data.DataLoader.__init__
    )

    # torch.profiler
    torch.profiler.ProfilerActivity.CUDA = torch.profiler.ProfilerActivity.MLU
    torch.profiler._KinetoProfile.prepare_trace = prepare_trace
    replace_profiler(torch_mlu.profiler.profiler_util, "_build_table")
    replace_profiler(torch.autograd.profiler.profile, "__init__")
    replace_profiler_legacy(torch.autograd.profiler_legacy.profile, "__init__")

    # storage
    torch.TypedStorage.cuda = torch.TypedStorage.mlu
    torch.TypedStorage.is_cuda = torch.TypedStorage.is_mlu

    torch.jit.script = script

    torch.cuda.nvtx = torch.mlu.cnpx
    torch.autograd.profiler.emit_nvtx = torch.autograd.profiler.emit_cnpx

    # cuda default
    torch.UntypedStorage._release_ipc_counter_cuda = (
        torch.mlu.UntypedStorage._release_ipc_counter_mlu
    )
    torch.UntypedStorage._share_cuda_ = torch.mlu.UntypedStorage._share_mlu_
    torch.UntypedStorage._new_shared_cuda = torch.mlu.UntypedStorage._new_shared_mlu
    # Regardless of whether the device parameter is cuda or other,
    # torch.TypedStorage._release_ipc_counter will call cuda's API.
    # Here it is replaced with torch.TypedStorage._release_ipc_counter_mlu,
    # and the mlu API will be called anyway.
    torch.TypedStorage._release_ipc_counter = (
        torch.TypedStorage._release_ipc_counter_mlu
    )
    torch.TypedStorage._share_cuda_ = torch.TypedStorage._share_mlu_
    torch.TypedStorage._new_shared_cuda = torch.TypedStorage._new_shared_mlu

    for func_name in default_cuda_args_list:
        replace_default_cuda_args(func_name)
