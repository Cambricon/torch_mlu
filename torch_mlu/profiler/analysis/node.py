# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
import torch
from abc import ABC
from typing import List, Optional, Tuple, Dict
from enum import IntEnum

from .common import utils
from .trace import (
    DurationEvent,
    EventTypes,
    KernelEvent,
    ModuleEvent,
    OperatorEvent,
    PLProfileEvent,
    CnclOpNameSet,
    NcclOpNameSet,
    GlooOpNameSet,
)

logger = utils.get_logger()

ExcludeOpName = ["DataParallel.forward", "DistributedDataParallel.forward"]


class BaseNode(ABC):
    def __init__(
        self,
        name: str,
        start_time: float,
        end_time: float,
        type: str,
        tid: int,
        external_id: Optional[int] = None,
        correlation_id: Optional[int] = None,
    ):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.tid = tid
        self.external_id = external_id  # For consistency check.
        self.correlation_id = correlation_id

    @staticmethod
    def get_node_argument(event: DurationEvent):
        kwargs = {}
        kwargs["name"] = event.name
        kwargs["start_time"] = event.ts
        kwargs["end_time"] = event.ts + event.duration
        kwargs["type"] = event.type
        kwargs["tid"] = event.tid

        external_id = getattr(event, "external_id", None)
        if external_id is not None:
            kwargs["external_id"] = external_id
        correlation_id = getattr(event, "correlation_id", None)
        if correlation_id is not None:
            kwargs["correlation_id"] = correlation_id

        return kwargs

    @property
    def duration(self) -> float:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return 0


class HostNode(BaseNode):
    def __init__(self, device_duration: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.device_duration = device_duration  # Total time of Kernel, GPU Memcpy, GPU Memset. TODO: parallel multi-stream? # noqa: E501


class OperatorNode(HostNode):
    # Don't use [] as default parameters
    # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument?page=1&tab=votes#tab-top
    # https://web.archive.org/web/20200221224620/http://effbot.org/zone/default-values.htm
    def __init__(
        self,
        children=None,
        runtimes=None,
        input_shape: Optional[List[List[int]]] = None,
        input_type: Optional[List[str]] = None,
        callstack: Optional[str] = None,
        self_host_duration: float = 0,
        self_device_duration: float = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.children: List[OperatorNode] = (
            [] if children is None else children
        )  # OperatorNode and ProfilerStepNode.
        self.runtimes: List[RuntimeNode] = (
            [] if runtimes is None else runtimes
        )  # RuntimeNode
        self.input_shape = input_shape
        self.input_type = input_type
        self.callstack = callstack
        self.self_host_duration = self_host_duration
        self.self_device_duration = self_device_duration

    @classmethod
    def header(cls):
        return [
            "Thread Id",
            "Name",
            "Input Shapes",
            "Input Type",
            "Call Stack",
            "Host Self Duration(us)",
            "Host Total Duration(us)",
            "Device Self Duration(us)",
            "Device Total Duration(us)",
        ]

    def data(self):
        return [
            self.tid,
            self.name,
            self.input_shape,
            self.input_type,
            self.callstack,
            round(self.self_host_duration, 3),
            round(self.end_time - self.start_time, 3),
            round(self.self_device_duration, 3),
            round(self.device_duration, 3),
        ]

    # Merge MLU overlapping RuntimeNodes for calculating self_host_duration.
    # Cnnl api will call cnrt api, so cnnl api duration contain cnrt's.
    # self_host_duration only need to exclude runtime api once.
    def _merge_runtimes(self):
        merged_runtimes = []
        for rt in self.runtimes:
            if rt.name == "dummy":
                continue
            if not merged_runtimes:
                merged_runtimes.append(rt)
            else:
                if (
                    rt.start_time >= merged_runtimes[-1].start_time
                    and rt.end_time <= merged_runtimes[-1].end_time
                ):
                    # Skip contained runtime api.
                    continue
                else:
                    merged_runtimes.append(rt)
        return merged_runtimes

    def fill_stats(self):
        # TODO: Replace recursive by using a stack, in case of too deep callstack.
        self.children.sort(key=lambda x: (x.start_time, -x.end_time))
        self.runtimes.sort(
            key=lambda x: (x.start_time, -x.end_time)
            if x.start_time and x.end_time
            else (sys.maxsize, -sys.maxsize - 1)
        )

        for child in self.children:
            child.fill_stats()
        for rt in self.runtimes:
            rt.fill_stats(self)

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= child.end_time - child.start_time

        for rt in self.runtimes:
            self.device_duration += rt.device_duration
            self.self_device_duration += rt.device_duration

        for rt in self._merge_runtimes():
            # From PyTorch 1.8 RC1, cpu_self_time does not include runtime's time.
            # So here we keep consistent with it.
            if rt.end_time is not None and rt.start_time is not None:
                self.self_host_duration -= rt.end_time - rt.start_time

    def get_operator_and_kernels(self):
        ops: List[OperatorNode] = []
        kernels: List[DeviceNode] = []
        for child in self.children:
            child_ops, child_kernels = child.get_operator_and_kernels()
            ops.extend(child_ops)
            kernels.extend(child_kernels)
        for rt in self.runtimes:
            kernels.extend(list(rt.get_kernels()))

        if is_operator_node(self):
            ops.append(self)

        return ops, kernels

    @classmethod
    def create(cls, event: OperatorEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(
            input_shape=event.input_shape,
            input_type=event.input_type,
            callstack=event.callstack,
            **kwargs
        )


class ProfilerStepNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ModuleNode(OperatorNode):
    def __init__(self, module_id: int, python_id: int, python_parent_id: int, **kwargs):
        super().__init__(**kwargs)
        self.module_id = module_id
        self.python_id = python_id
        self.python_parent_id = python_parent_id

    def fill_stats(self):
        super().fill_stats()
        self.self_device_duration += get_chilren_self_device_time(self)

    @classmethod
    def create(cls, event: ModuleEvent):
        kwargs = BaseNode.get_node_argument(event)
        kwargs["module_id"] = event.module_id
        kwargs["python_id"] = event.python_id
        kwargs["python_parent_id"] = event.python_parent_id
        # From the time being, the ModuleNode always have external_id to 0.
        # As the result, we need reset the external_id to None to ignore adding the runtime nodes for ModuleNode
        kwargs.pop("external_id", None)
        return cls(**kwargs)


class BackwardNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fill_stats(self):
        """Override the timestamps and duration for BackwardNode only"""
        self.children.sort(key=lambda x: (x.start_time, -x.end_time))
        self.start_time = self.children[0].start_time
        self.end_time = self.children[-1].end_time

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= child.end_time - child.start_time


class PLProfileNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, event: PLProfileEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(**kwargs)


class PLModuleNode(OperatorNode):
    def __init__(self, module_id: int, **kwargs):
        super().__init__(**kwargs)
        self.module_id = module_id

    def fill_stats(self):
        super().fill_stats()
        self.self_device_duration += get_chilren_self_device_time(self)

    @classmethod
    def create(cls, event: PLProfileEvent):
        kwargs = BaseNode.get_node_argument(event)
        kwargs["module_id"] = event.module_id
        return cls(**kwargs)


class DataLoaderNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OptimizerNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RuntimeNode(HostNode):
    def __init__(
        self,
        device_nodes: Optional[List["DeviceNode"]] = None,
        parent_rt_node=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # One runtime could trigger more than one kernel, such as cudaLaunchCooperativeKernelMultiDevice.
        self.device_nodes = (
            sorted(device_nodes, key=lambda x: (x.start_time, -x.end_time))
            if device_nodes
            else None
        )
        # add this member just for use of RuntimeNodes with external_id=0, in order to
        # show kernel's op_node name by the topmost rt_node name rather than 'CallTreeRoot'.
        self.parent_rt_node = parent_rt_node

    def fill_stats(self, op_node: OperatorNode = None):
        if self.device_nodes:
            for device_node in self.device_nodes:
                if op_node:
                    if op_node.name == "CallTreeRoot":
                        if device_node.tasktopo_external_op:
                            # for MLUGraph kernels, use op name in capture stage
                            device_node.op_name = device_node.tasktopo_external_op
                        else:
                            # use the topmost runtime node's name to instead 'CallTreeRoot'
                            device_node.op_name = self.name
                            parent = self.parent_rt_node
                            while parent is not None:
                                if parent.name == "DummyZeroRuntimeRoot":
                                    break
                                device_node.op_name = parent.name
                                parent = parent.parent_rt_node
                    else:
                        device_node.op_name = op_node.name
                device_duration = device_node.end_time - device_node.start_time
                self.device_duration += device_duration

    def get_kernels(self):
        if self.device_nodes:
            for d in self.device_nodes:
                if d.type == EventTypes.KERNEL:
                    yield d

    @classmethod
    def create(cls, event, device_nodes: Optional[List["DeviceNode"]]):
        kwargs = BaseNode.get_node_argument(event)
        return cls(device_nodes=device_nodes, **kwargs)


class CounterType(IntEnum):
    IPU = 0
    MEMCORE = 1
    BANDWIDTH = 2


class DeviceUtils:
    def __init__(
        self, header: str = None, counter: str = None, counter_type: CounterType = None
    ):
        self.header: str = header
        self.counter: str = counter
        self.counter_type: CounterType = counter_type


class DeviceNode(BaseNode):
    pmu_header: List[str] = []
    utils_list: List[DeviceUtils] = []
    utils_header: List[str] = []
    total_header: List[str] = []
    header_init: bool = False

    frequency_map = {}
    bandwidth_map = {}
    core_count_map = {}

    def __init__(
        self,
        device_id: int = None,
        stream_id: int = None,
        context_id: int = None,
        kernel_type: str = None,
        dim: Optional[List[int]] = None,
        tasktopo: int = None,
        tasktopo_node: int = None,
        tasktopo_external_op: str = None,
        pmus: Dict = {},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.op_name = None
        self.device_id = device_id
        self.stream_id = stream_id
        self.context_id = context_id
        self.kernel_type = kernel_type
        self.dim = dim
        self.tasktopo = tasktopo
        self.tasktopo_node = tasktopo_node
        self.tasktopo_external_op = tasktopo_external_op
        self.pmus_data = []
        self.utils_data = []
        self.total_data = []

        if pmus:
            if not DeviceNode.header_init:
                DeviceNode.utils_list = [
                    DeviceUtils("lt_utils(%)", "tp_core__lt_cycles", CounterType.IPU),
                    DeviceUtils(
                        "ct_utils(%)", "tp_core__csimd_post_cycles", CounterType.IPU
                    ),
                    DeviceUtils(
                        "dram_read_utils(%)",
                        "tp_core__dram_read_cycles",
                        CounterType.IPU,
                    ),
                    DeviceUtils(
                        "dram_write_utils(%)",
                        "tp_core__dram_write_cycles",
                        CounterType.IPU,
                    ),
                    DeviceUtils(
                        "bandwidth_read_utils(%)",
                        "tp_cluster__read_bytes",
                        CounterType.BANDWIDTH,
                    ),
                    DeviceUtils(
                        "bandwidth_write_utils(%)",
                        "tp_cluster__write_bytes",
                        CounterType.BANDWIDTH,
                    ),
                    DeviceUtils(
                        "memcore_io_read_utils(%)",
                        "tp_memcore__dram_read_cycles",
                        CounterType.MEMCORE,
                    ),
                    DeviceUtils(
                        "memcore_io_write_utils(%)",
                        "tp_memcore__dram_write_cycles",
                        CounterType.MEMCORE,
                    ),
                    DeviceUtils(
                        "mv_utils(%)", "tp_core__mv_inst_cycles", CounterType.IPU
                    ),
                    DeviceUtils("alu_utils(%)", "tp_core__alu_cycles", CounterType.IPU),
                ]
                DeviceNode.utils_header = [
                    utils.header for utils in DeviceNode.utils_list
                ]
                DeviceNode.total_header = [
                    "total_approximate_bandwidth_bytes",
                    "total_approximate_ipu_cycles",
                    "total_approximate_memcore_cycles",
                ]
                DeviceNode.pmu_header = list(pmus.keys())
                DeviceNode.header_init = True

            if device_id not in DeviceNode.frequency_map:
                device_prop = torch.mlu.get_device_properties(device_id)
                DeviceNode.frequency_map[device_id] = device_prop.ipu_frequency
                DeviceNode.bandwidth_map[device_id] = device_prop.dram_bandwidth
                DeviceNode.core_count_map[device_id] = device_prop.multi_processor_count
            frequency = DeviceNode.frequency_map.get(device_id)  # MHz
            bandwidth = DeviceNode.bandwidth_map.get(device_id)  # Bytes/us
            core_count = DeviceNode.core_count_map.get(device_id)
            duration = self.end_time - self.start_time  # us
            total_cycles_approximation = frequency * duration * core_count
            total_bandwidth_approximation = bandwidth * duration

            get_utils = lambda x, y: round(x / y * 100, 3) if y != 0 else None

            for utils in DeviceNode.utils_list:
                if utils.counter in pmus:
                    total = 0
                    if utils.counter_type is CounterType.IPU:
                        total = total_cycles_approximation
                    elif utils.counter_type is CounterType.MEMCORE:
                        total = total_cycles_approximation / 4
                    elif utils.counter_type is CounterType.BANDWIDTH:
                        total = total_bandwidth_approximation
                    self.utils_data.append(get_utils(pmus[utils.counter], total))
                else:
                    self.utils_data.append("N/A")

            self.total_data = [
                round(total_bandwidth_approximation, 3),
                round(total_cycles_approximation, 3),
                round(total_cycles_approximation / 4, 3),
            ]
            self.pmus_data = list(pmus.values())

        # for l2_cache.csv
        self.llc_total = pmus.get("llc__tagram_hit", 0) + pmus.get(
            "llc__tagram_miss", 0
        )
        self.hit_rate = (
            round(pmus.get("llc__tagram_hit") / self.llc_total, 3)
            if self.llc_total
            else "N/A"
        )
        self.viction = pmus.get("llc__eviction", "N/A")

    @classmethod
    def header(cls):
        return (
            [
                "Thread Id",
                "Correlation Id",
                "Kernel Name",
                "Operator",
                "Start Time",
                "Duration(us)",
                "External Id",
                "Device Id",
                "Stream Id",
                "Context Id",
                "Kernel Type",
                "Dims",
                "Tasktopo",
                "Tasktopo Node",
            ]
            + DeviceNode.utils_header
            + DeviceNode.total_header
            + DeviceNode.pmu_header
        )

    def data(self):
        return (
            [
                self.tid,
                self.correlation_id,
                self.name,
                self.op_name,
                self.start_time,
                round(self.end_time - self.start_time, 3),
                self.external_id,
                self.device_id,
                self.stream_id,
                self.context_id,
                self.kernel_type,
                self.dim,
                self.tasktopo,
                self.tasktopo_node,
            ]
            + self.utils_data
            + self.total_data
            + self.pmus_data
        )

    @classmethod
    def l2_cache_header(cls):
        return [
            "Thread Id",
            "Kernel Name",
            "Stream Id",
            "Correlation Id",
            "Hit Rate",
            "llc__eviction",
        ]

    def l2_cache_data(self):
        return [
            self.tid,
            self.name,
            self.stream_id,
            self.correlation_id,
            self.hit_rate,
            self.viction,
        ]

    @classmethod
    def create(cls, event: KernelEvent):
        kwargs = BaseNode.get_node_argument(event)
        if event.type == EventTypes.KERNEL:
            kwargs["device_id"] = event.device_id
            kwargs["stream_id"] = event.stream_id
            kwargs["context_id"] = event.context_id
            kwargs["kernel_type"] = event.kernel_type
            kwargs["dim"] = event.dim
            kwargs["tasktopo"] = event.tasktopo
            kwargs["tasktopo_node"] = event.tasktopo_node
            kwargs["tasktopo_external_op"] = event.tasktopo_external_op
            kwargs["pmus"] = event.pmus
        return cls(**kwargs)


def create_operator_node(event: OperatorEvent):
    if (
        event.name.startswith("enumerate(DataLoader)#")
        and event.name.endswith(".__next__")
        or event.name.startswith("enumerate(DataPipe)#")
    ):
        return DataLoaderNode.create(event)
    elif event.name.startswith("Optimizer.step"):
        return OptimizerNode.create(event)
    elif event.type == EventTypes.USER_ANNOTATION:
        # USER_ANNOTATION is just a label, can't regard as OperatorNode.
        return None
    else:
        return OperatorNode.create(event)


def is_operator_node(node: BaseNode):
    return bool(
        type(node) is OperatorNode
        and node.type == EventTypes.OPERATOR
        and node.name not in ExcludeOpName
        and not node.name.startswith("Optimizer.")
    )  # exclude Optimizer.zero_grad


def get_chilren_self_device_time(node):
    self_device_duration = 0
    for child in node.children:
        if is_operator_node(child):
            self_device_duration += child.device_duration
    return self_device_duration
