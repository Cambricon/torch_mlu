from collections import Counter
from typing import Callable, Dict, List

from .common import utils
from .node import OperatorNode

logger = utils.get_logger()


class OperatorAgg:
    def __init__(self, op: OperatorNode):
        self.name = op.name
        self.input_shape = str(op.input_shape)  # Optional

        self.callstacks = set()  # Optional
        self.calls: int = 0
        self.host_duration: float = 0
        self.device_duration: float = 0
        self.self_host_duration: float = 0
        self.self_device_duration: float = 0
        # aggregate detailed information of op's launch kernels
        self.kernel_calls: int = 0
        self.kernel_dict = Counter()


def aggregate_ops(
    op_list: List[OperatorNode], keys_func: List[Callable[[OperatorNode], str]]
) -> List[Dict[str, OperatorAgg]]:
    def aggregate(key_to_agg: Dict[str, OperatorAgg], key: str, op: OperatorNode):
        if key not in key_to_agg:
            key_to_agg[key] = OperatorAgg(op)
        agg = key_to_agg[key]
        agg.callstacks.add(op.callstack)
        agg.calls += 1
        agg.host_duration += op.duration
        agg.device_duration += op.device_duration
        agg.self_host_duration += op.self_host_duration
        agg.self_device_duration += op.self_device_duration

        _, kernels = op.get_operator_and_kernels()
        agg.kernel_calls += len(kernels)
        kernels_cnter = Counter([kernel.name for kernel in kernels])
        agg.kernel_dict.update(kernels_cnter)

        return agg

    agg_dicts: List[Dict[str, OperatorAgg]] = [{} for _ in range(len(keys_func))]
    for op in op_list:
        for i, key_func in enumerate(keys_func):
            key = key_func(op)
            aggregate(agg_dicts[i], key, op)

    return agg_dicts
