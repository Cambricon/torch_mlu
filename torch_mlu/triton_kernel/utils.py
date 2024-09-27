import functools
import torch
import torch_mlu


@functools.lru_cache(1)
def get_total_core_num():
    if torch.mlu.is_available():
        device_prop = torch.mlu.get_device_properties(torch.mlu.current_device())
        total_cluster_num = device_prop.cluster_count
        total_core_num = total_cluster_num * device_prop.core_num_per_cluster
        return total_core_num
    return 0
