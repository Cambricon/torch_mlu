from typing import Dict, List, Optional

import sys
import pandas as pd

from .trace import EventTypes
from .node import DeviceNode


class KernelAggByNameOp:
    def __init__(self, kernel: DeviceNode, op_name: str):
        self.name = kernel.name
        self.op_name = op_name
        # for mlu
        self.kernel_type = kernel.kernel_type
        self.dim = kernel.dim

        self.calls: int = 0
        self.total_duration: float = 0
        self.min_duration: float = sys.float_info.max
        self.max_duration: float = 0

    @property
    def avg_duration(self):
        return self.total_duration / self.calls


def aggregate_kernels(kernel_list: List[DeviceNode]) -> List[KernelAggByNameOp]:
    name_op_to_agg: Dict[str, KernelAggByNameOp] = {}
    for kernel in kernel_list:
        dur = kernel.end_time - kernel.start_time
        op_name = "N/A" if kernel.op_name is None else kernel.op_name
        key = "###".join((kernel.name, op_name))
        if key not in name_op_to_agg:
            name_op_to_agg[key] = KernelAggByNameOp(kernel, op_name)
        agg = name_op_to_agg[key]
        agg.calls += 1
        agg.total_duration += dur
        agg.min_duration = min(agg.min_duration, dur)
        agg.max_duration = max(agg.max_duration, dur)

    kernel_list_groupby_name_op = list(name_op_to_agg.values())
    return kernel_list_groupby_name_op


class KernelParser:
    def __init__(self):
        self.kernel_stat: Optional[pd.DataFrame] = None
        self.kernel_header = [
            "Kernel Name",
            "Count",
            "Total Time(us)",
            "Min Time(us)",
            "Avg Time(us)",
            "Max Time(us)",
            "Ratio(%)",
        ]
        self.kernel_header_key = ["count", "sum", "min", "mean", "max", "ratio"]
        self.kernel_sum_total = 0

        self.kernel_list_groupby_name_op: List[KernelAggByNameOp] = None
        self.op_kernel_header = [
            "Kernel Name",
            "Operator",
            "Count",
            "Total Time(us)",
            "Min Time(us)",
            "Avg Time(us)",
            "Max Time(us)",
            "Ratio(%)",
        ]

    def parse_events(self, events, kernels):
        events = [vars(event) for event in events if event.type == EventTypes.KERNEL]
        events = pd.DataFrame(events)
        events = events.astype({"type": "category", "name": "string"}, copy=False)

        self.kernel_stat = (
            events.groupby("name")
            .agg(
                count=("duration", "count"),
                sum=("duration", "sum"),
                mean=("duration", "mean"),
                max=("duration", "max"),
                min=("duration", "min"),
            )
            .sort_values("sum", ascending=False)
        )
        self.kernel_sum_total = self.kernel_stat["sum"].sum()
        self.kernel_stat["ratio"] = (
            self.kernel_stat["sum"] / self.kernel_sum_total * 100.0
        )

        self.kernel_list_groupby_name_op = aggregate_kernels(kernels)

    def get_kernel_statistic(self):
        kernel_rows = []
        for _id, (name, row) in enumerate(self.kernel_stat.iterrows()):
            kernel_row = [name]
            for key in self.kernel_header_key:
                kernel_row.append(row[key])
            kernel_rows.append(kernel_row)
        return kernel_rows

    def get_op_kernel_statistic(self):
        kernel_list: List[KernelAggByNameOp] = sorted(
            self.kernel_list_groupby_name_op,
            key=lambda x: x.total_duration,
            reverse=True,
        )
        op_kernel_rows = []
        for agg_by_name_op in kernel_list:
            op_kernel_row = [
                agg_by_name_op.name,
                agg_by_name_op.op_name,
                agg_by_name_op.calls,
                agg_by_name_op.total_duration,
                agg_by_name_op.min_duration,
                agg_by_name_op.avg_duration,
                agg_by_name_op.max_duration,
                agg_by_name_op.total_duration / self.kernel_sum_total * 100.0,
            ]
            op_kernel_rows.append(op_kernel_row)

        return op_kernel_rows
