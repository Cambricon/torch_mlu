import os
import warnings


mlu_env_map = {
    # PyTorch Environment Variables
    "PYTORCH_NO_CUDA_MEMORY_CACHING": "PYTORCH_NO_MLU_MEMORY_CACHING",
    "PYTORCH_CUDA_ALLOC_CONF": "PYTORCH_MLU_ALLOC_CONF",
    "PYTORCH_NVML_BASED_CUDA_CHECK": "PYTORCH_CNDEV_BASED_MLU_CHECK",
    "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "",
    "TORCH_CUDNN_V8_API_DISABLED": "",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE",
    "TORCH_CUDNN_V8_API_DEBUG": "",
    # CUDA Runtime and Libraries Environment Variables
    "CUDA_VISIBLE_DEVICES": "MLU_VISIBLE_DEVICES",
    "CUDA_LAUNCH_BLOCKING": "",
    "CUBLAS_WORKSPACE_CONFIG": "",
    "CUDNN_CONV_WSCAP_DBG": "",
    "CUBLASLT_WORKSPACE_SIZE": "",
    "CUDNN_ERRATA_JSON_FILE": "",
    "NVIDIA_TF32_OVERRIDE": "",
    # NCCL environment variables
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "TORCH_CNCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "TORCH_CNCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_BLOCKING_WAIT": "TORCH_CNCL_BLOCKING_WAIT",
    "TORCH_NCCL_COORD_CHECK_MILSEC": "TORCH_CNCL_COORD_CHECK_MILSEC",
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": "TORCH_CNCL_DEBUG_INFO_PIPE_FILE",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "TORCH_CNCL_DEBUG_INFO_TEMP_FILE",
    "TORCH_NCCL_DESYNC_DEBUG": "TORCH_CNCL_DESYNC_DEBUG",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "TORCH_CNCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_ENABLE_MONITORING": "TORCH_CNCL_ENABLE_MONITORING",
    "TORCH_NCCL_ENABLE_TIMING": "TORCH_CNCL_ENABLE_TIMING",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCH_NCCL_HIGH_PRIORITY": "",
    "TORCH_NCCL_NAN_CHECK": "",
    "TORCH_NCCL_NONBLOCKING_TIMEOUT": "",
    "TORCH_NCCL_RETHROW_CUDA_ERRORS": "",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "TORCH_CNCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_TRACE_CPP_STACK": "TORCH_CNCL_TRACE_CPP_STACK",
    "TORCH_NCCL_USE_COMM_NONBLOCKING": "",
    "TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK": "",
    "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC": "TORCH_CNCL_WAIT_TIMEOUT_DUMP_MILSEC",
    "NCCL_ASYNC_ERROR_HANDLING": "TORCH_CNCL_ASYNC_ERROR_HANDLING",
    "NCCL_BLOCKING_WAIT": "TORCH_CNCL_BLOCKING_WAIT",
    "NCCL_DESYNC_DEBUG": "TORCH_CNCL_DESYNC_DEBUG",
    "NCCL_ENABLE_TIMING": "TORCH_CNCL_ENABLE_TIMING",
    "NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK": "",
}

# Custom Environment Variable Retrieval Blacklist
#
# The implementation of CUDA environment variable compatibility in torch_mlu
# is based on two key principles:
#
# 1. On the C++ side, implemented in the torch_mlu csrc directory,
# CUDA-compatible environments are accessed using the following three functions:
# getCvarString, getCvarInt, and getCvarBool. This ensures that both CUDA
# and MLU environment variables are handled simultaneously in the csrc backend.
#
# 2. On the Python side, implemented in the torch_mlu mlu directory,
# CUDA-compatible environment variables are retrieved using getenv.
#
# The environment variable retrieval logic in torch_mlu follows these two principles.
#
# Therefore, if you define a custom MLU environment variable using
# getCvarString, getCvarInt, or getCvarBool on the C++ side, or if you use
# getenv on the Python side, and the environment variable name contains any of
# the following keywords: ["TORCH_CNCL", "MLU", "CAMBRICON", "CNMATMUL"], you
# must add it to the Custom Environment Variable Retrieval Blacklist. This ensures
# that it is excluded when retrieving CUDA-compatible environment variables.
mlu_env_blacklist = [
    "",
]


# Check for unsupported Pytorch CUDA environment variables.
def unsupport_env_check():
    unsupport_env_list = [
        cuda_env
        for cuda_env, mlu_env in mlu_env_map.items()
        if cuda_env in os.environ and not mlu_env
    ]
    if unsupport_env_list:
        formatted_envs = ", ".join(f"'{env}'" for env in unsupport_env_list)
        warnings.warn(
            f"The following PyTorch environment variables are enabled, but they are not yet supported by MLU: {formatted_envs}. "
            "Please refer to the documentation to see which PyTorch environment variables are supported by MLU."
        )
