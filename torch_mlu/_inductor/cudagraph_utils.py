from typing import Dict, Optional

import torch
from torch._inductor.cudagraph_utils import (
    format_default_skip_message,
    get_use_stack_trace,
)

from ..mlu._utils import replace_references


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        if stack_trace := get_use_stack_trace(cpu_node):
            return format_default_skip_message(
                f"cpu device. Found from : \n {stack_trace}"
            )

        return format_default_skip_message("cpu device")

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "mlu"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping.keys())
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


replace_references(
    torch._inductor.cudagraph_utils.check_multiple_devices_or_any_cpu_nodes,
    check_multiple_devices_or_any_cpu_nodes,
)
