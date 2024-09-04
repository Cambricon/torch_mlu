import os
import warnings

mlu_env_map = {
    # PyTorch Environment Variables
    "PYTORCH_CUDA_ALLOC_CONF": "PYTORCH_MLU_ALLOC_CONF",
    "PYTORCH_CUDA_DSA_STACKTRACING": "",
    "PYTORCH_NO_CUDA_MEMORY_CACHING": "PYTORCH_NO_MLU_MEMORY_CACHING",
    "PYTORCH_NVML_BASED_CUDA_CHECK": "",
    "PYTORCH_USE_CUDA_DSA": "",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE",
    "TORCH_CUDNN_USE_HEURISTIC_MODE_B": "",
    "TORCH_CUDNN_V8_API_DEBUG": "",
    "TORCH_CUDNN_V8_API_DISABLED": "",
    "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "",
    "TORCH_LINALG_PREFER_CUSOLVER": "",
    # CUDA Runtime and Libraries Environment Variables
    "CUBLASLT_WORKSPACE_SIZE": "",
    "CUBLAS_WORKSPACE_CONFIG": "",
    "CUDA_LAUNCH_BLOCKING": "",
    "CUDA_MODULE_LOADING": "",
    "CUDA_VISIBLE_DEVICES": "",
    "DISABLE_ADDMM_CUDA_LT": "",
    # NCCL environment variables
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "TORCH_CNCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_NONBLOCKING_TIMEOUT": "",
    "TORCH_NCCL_USE_COMM_NONBLOCKING": "",
    "NCCL_AVOID_RECORD_STREAMS": "TORCH_CNCL_AVOID_RECORD_STREAMS",
}


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
