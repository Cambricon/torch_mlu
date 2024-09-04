from functools import wraps
import functools
from typing import Callable, Dict, List, Tuple, Union
import warnings
import functools
import sys

import torch
import torch.autograd.profiler as prof
import torch.distributed.distributed_c10d as c10d
import torch.distributed.fsdp
import torch.utils._device

import torch_mlu

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
    torch.amp.GradScaler,
    torch.mlu.device,
    torch.storage.UntypedStorage,
    torch.UntypedStorage,
    torch.TypedStorage,
    # fsdp
    torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel,
    torch.utils._device.DeviceContext,
]


# torch.* that needs to replace device, selected from torch/_torch_docs.py
torch_fn_list = [
    "set_default_device",
    "as_tensor",
    "asarray",
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
    "empty_permuted",
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
    "Stream",
    "Event",
    "get_autocast_dtype",
    "set_autocast_dtype",
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
    "type",
    "mlu",
]

module_fn_list = [
    "to",
    "to_empty",
]

torch_mlu_memory_fn_list = [
    "mem_get_info",
]


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
            for key in [
                "device",
                "device_type",
                "output_device",
                "device_id",
                "map_location",
            ]:
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
    elif isinstance(arg, str) and "CUDA" in arg:
        arg = arg.replace("CUDA", "MLU")
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
# 3. backend='cuda:nccl' -> backend='mlu:cncl'
def replace_nccl_with_cncl(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, str) and "nccl" in arg:
                    args[i] = arg.replace("nccl", "cncl")
                    if "cuda" in args[i]:
                        args[i] = args[i].replace("cuda", "mlu")
                if isinstance(arg, str) and "NCCL" in arg:
                    args[i] = arg.replace("NCCL", "CNCL")
        if kwargs:
            backend = kwargs.get("backend", None)
            if isinstance(backend, str) and "nccl" in backend:
                backend = backend.replace("nccl", "cncl")
                if "cuda" in backend:
                    backend = backend.replace("cuda", "mlu")
                kwargs["backend"] = backend
        return fn(*args, **kwargs)

    return wrapper_fn


# profiler
# sort_by='*cuda*' -> sort_by='*mlu*'
def replace_profiler_args(fn):
    @wraps(fn)
    def wrapper_fn(*args, **kwargs):
        if args:
            args = list(args)
            for i, arg in enumerate(args):
                args[i] = replace_cuda_with_mlu(arg)

        if kwargs:
            activities = kwargs.get("activities", None)
            if activities and torch.profiler.ProfilerActivity.CUDA in activities:
                activities.remove(torch.profiler.ProfilerActivity.CUDA)
                activities.append(torch.profiler.ProfilerActivity.PrivateUse1)
            sort_by = kwargs.get("sort_by", None)
            if sort_by:
                kwargs["sort_by"] = replace_cuda_with_mlu(sort_by)
            metric = kwargs.get("metric", None)
            if metric:
                kwargs["metric"] = replace_cuda_with_mlu(metric)
        return fn(*args, **kwargs)

    return wrapper_fn


def replace_profiler(module, fn_name):
    fn = getattr(module, fn_name, None)
    if fn:
        setattr(module, fn_name, replace_profiler_args(fn))


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


# For autocast API.
# set_autocast_enabled(device_type: str, enabled: _bool)
# set_autocast_enabled(enabled: _bool)
# This API is default for cuda, and will be deprecated in pytorch 2.5;
# is_autocast_enabled()
# is_autocast_enabled(device_type: str)
# replace_autocast function:
# replace torch.is_autocast_enabled() to torch.is_autocast_enabled('mlu')
# torch.is_autocast_enabled('cuda') to torch.is_autocast_enabled('mlu')
# torch.set_autocast_enabled(bool) to torch.set_autocast_enabled('mlu', bool)
# torch.set_autocast_enabled('cuda', bool) to torch.set_autocast_enabled('mlu', bool)
def replace_autocast(fn):
    @wraps(fn)
    def warp_fn(*args, **kwargs):
        parameters_len = len(args) + len(kwargs)
        if parameters_len == 0:
            # is_autocast_enabled()
            args = [torch._C._get_privateuse1_backend_name()]
            return fn(*args, **kwargs)
        elif parameters_len == 1:
            # for set_autocast_enabled(enabled: _bool)
            if (
                len(args) == 1
                and isinstance(args[0], bool)
                or len(kwargs) == 1
                and "enabled" in kwargs.keys()
            ):
                return fn(torch._C._get_privateuse1_backend_name(), *args, **kwargs)
            else:
                # for is_autocast_enabled(device_type: str)
                return replace_device_args(fn)(*args, **kwargs)
        else:
            # set_autocast_enabled(device_type: str, enabled: _bool)
            return replace_device_args(fn)(*args, **kwargs)

    return warp_fn


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


old_device_constructors_ = torch.utils._device._device_constructors()


@functools.lru_cache(1)
def original_device_constructors():
    global old_device_constructors_
    return old_device_constructors_


def _new_privateuse1_deserialize(obj, location):
    if location.startswith("cuda"):
        location = location.replace("cuda", "mlu")
    origin_privateuse1_deserialize = functools.partial(
        torch.serialization._deserialize, "privateuse1"
    )
    return origin_privateuse1_deserialize(obj, location)


torch.serialization.register_package(
    11,
    functools.partial(torch.serialization._backend_tag, "privateuse1"),
    _new_privateuse1_deserialize,
)


def apply_monkey_patches():
    # replace order MATTERS, this functions need to be replaced before torch_fn_list
    torch.utils._device._device_constructors = original_device_constructors
    # torch.*
    replace_device(torch, torch_fn_list)
    torch.get_autocast_gpu_dtype = functools.partial(
        torch.get_autocast_dtype, torch._C._get_privateuse1_backend_name()
    )
    torch.set_autocast_gpu_dtype = functools.partial(
        torch.set_autocast_dtype, torch._C._get_privateuse1_backend_name()
    )
    torch.is_autocast_enabled = replace_autocast(torch.is_autocast_enabled)
    torch.set_autocast_enabled = replace_autocast(torch.set_autocast_enabled)
    torch.amp.custom_fwd = replace_autocast(torch.amp.custom_fwd)
    torch.amp.custom_bwd = replace_autocast(torch.amp.custom_bwd)
    torch._C._cuda_getCurrentRawStream = torch_mlu._MLUC._mlu_getCurrentRawStream
    torch._C._cuda_setCheckpointPoolState = torch_mlu._MLUC._mlu_setCheckpointPoolState
    torch._C._cuda_getCheckpointState = torch_mlu._MLUC._mlu_getCheckpointState
    torch._C._cuda_cudaCachingAllocator_raw_delete = (
        torch_mlu._MLUC._mlu_mluCachingAllocator_raw_delete
    )
    torch._C._cuda_checkPoolLiveAllocations = (
        torch_mlu._MLUC._mlu_checkPoolLiveAllocations
    )
    torch._C._cuda_beginAllocateCurrentStreamToPool = (
        torch_mlu._MLUC._mlu_beginAllocateCurrentStreamToPool
    )
    torch._C._cuda_endAllocateCurrentStreamToPool = (
        torch_mlu._MLUC._mlu_endAllocateCurrentStreamToPool
    )
    torch._C._cuda_releasePool = torch_mlu._MLUC._mlu_releasePool

    # torch.cuda.*
    torch.cuda = torch.mlu
    sys.modules['torch.cuda'].__dict__['_lazy_call'] = torch.mlu._lazy_call
    torch.cuda.nccl = torch.mlu.cncl
    torch.cuda.CUDAGraph = torch.mlu.MLUGraph
    replace_device(torch.cuda, torch_cuda_fn_list)

    # torch.Tensor.*
    replace_device(torch.Tensor, tensor_fn_list)
    torch.Tensor.cuda = torch.Tensor.mlu
    torch.Tensor.is_cuda = torch.Tensor.is_mlu

    # torch.nn.Module.*
    replace_device(torch.nn.Module, module_fn_list)
    torch.nn.Module.cuda = torch.nn.Module.mlu

    replace_device(torch.mlu.memory, torch_mlu_memory_fn_list)
    replace_device(torch_mlu.mlu.storage, ["_typed_storage_init"])

    for mod in class_list:
        replace_device(mod, ["__init__"])

    # torch.Generator
    # can't set attributes of extension type 'torch._C.Generator',
    # so we use a subclass of Generator
    torch.Generator = Generator
    replace_device(torch.Generator, ["__new__"])
    replace_device(torch.UntypedStorage, ["__new__"])
    replace_device(torch.TypedStorage, ["__new__"])

    # torch.distributed.init_process_group
    c10d.init_process_group = replace_nccl_with_cncl(c10d.init_process_group)
    torch.distributed.init_process_group = replace_nccl_with_cncl(
        torch.distributed.init_process_group
    )
    torch.distributed.BackendConfig.__init__ = replace_nccl_with_cncl(
        torch.distributed.BackendConfig.__init__
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

    # TODO(PYTORCH-11776): A workround to skip ProcessGroupCudaP2P,
    # remove following code when MLU support ProcessGroupCudaP2P.
    c10d.ProcessGroupCudaP2P = torch_mlu._MLUC.ProcessGroupCNCL
    torch.distributed.ProcessGroupCudaP2P = torch_mlu._MLUC.ProcessGroupCNCL

    # If the backend is not provided, then both a gloo
    # and nccl backend will be created by init_process_group,
    # we create gloo and cncl instead of gloo and nccl.
    c10d.Backend.default_device_backend_map.pop("cuda", "nccl")
    torch._C._cuda_hasPrimaryContext = torch_mlu._MLUC._mlu_hasPrimaryContext

    # torch.utils.data.DataLoader
    torch.utils.data.DataLoader.__init__ = replace_dataloader(
        torch.utils.data.DataLoader.__init__
    )

    # torch.profiler
    replace_profiler(torch.profiler._KinetoProfile, "__init__")
    replace_profiler(torch.autograd.profiler_util, "_build_table")
    replace_profiler(torch.autograd.profiler_util.EventList, "export_stacks")

    # storage
    torch.TypedStorage.cuda = torch.TypedStorage.mlu
    torch.TypedStorage.is_cuda = torch.TypedStorage.is_mlu

    torch.jit.script = script

    torch.cuda.nvtx = torch.mlu.cnpx
    torch.autograd.profiler.emit_nvtx = torch.autograd.profiler.emit_cnpx

    torch.backends.cuda = torch.backends.mlu
