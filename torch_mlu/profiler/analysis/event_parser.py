# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from collections import defaultdict
from enum import IntEnum
from typing import Dict, Iterable, List

from .node import (
    DeviceNode,
    ModuleNode,
    OperatorNode,
    PLModuleNode,
    PLProfileNode,
    ProfilerStepNode,
    RuntimeNode,
    create_operator_node,
)
from .op_tree import OpTreeBuilder
from .trace import (
    BaseEvent,
    DurationEvent,
    EventTypes,
)
from .common import utils

logger = utils.get_logger()


class ProfileRole(IntEnum):
    Kernel = 0
    Memcpy = 1
    Memset = 2
    Communication = 3
    Runtime = 4
    DataLoader = 5
    CpuOp = 6
    Other = 7
    Total = 8


class NodeParserMixin:
    def __init__(self, *args, **kwargs):
        """Please refer to https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way # noqa: E501
        to see the reason why we need call super().__init__ like this way
        """
        super().__init__(*args, **kwargs)

    def parse_nodes(self, events: Iterable[BaseEvent]):
        # For OperatorNode and ProfilerStepNode:
        #   Use time interval containing relationship to build father-child correlation,
        #   which is consistent with autograd profiler.
        # For RuntimeNode:
        #   Use external_id to build correlation with its father OperatorNode or ProfilerStepNode.
        #   Because in the case when RuntimeNode has duration 0 and starts at same time as a OperatorNode,
        #   just use interval containing relationship can't tell it is child or brother of the OperatorNode.
        # value is a list of OperatorNode and ProfilerStepNode. Do not include RuntimeNode
        tid2list: Dict[int, List[OperatorNode]] = defaultdict(list)
        # value is  a list of PLProfileNode. Do not include RuntimeNode
        pl_tid2list: Dict[int, List[PLProfileNode]] = defaultdict(list)
        # value is a list of RuntimeNode with external_id=0. They will be attached to root nodes.
        tid2zero_rt_list: Dict[int, List[RuntimeNode]] = defaultdict(list)
        corrid_to_device: Dict[int, List[DeviceNode]] = defaultdict(
            list
        )  # value is a list of DeviceNode

        corrid_to_runtime: Dict[int, RuntimeNode] = {}  # value is a RuntimeNode
        externalid_to_runtime: Dict[int, List[RuntimeNode]] = defaultdict(
            list
        )  # value is a list of RuntimeNode

        for event in events:
            if event.type == EventTypes.MEMORY:
                continue
            self._parse_node(
                event,
                corrid_to_device,
                corrid_to_runtime,
                externalid_to_runtime,
                tid2list,
                pl_tid2list,
                tid2zero_rt_list,
            )

        # associate MLU Runtimes with CPU events
        for op_list in tid2list.values():
            for op in op_list:
                runtime_nodes = externalid_to_runtime.pop(op.external_id, [])
                if runtime_nodes:
                    op.runtimes.extend(runtime_nodes)
        for ext_id in externalid_to_runtime:
            if ext_id != 0:
                logger.debug(
                    "{} Runtime with external id {} don't correlate to any operator!".format(
                        len(externalid_to_runtime[ext_id]), ext_id
                    )
                )

        if len(corrid_to_device) > 0:
            node_count_dict = defaultdict(int)
            for nodes in corrid_to_device.values():
                for n in nodes:
                    node_count_dict[n.type] += 1
                    logger.debug(
                        (
                            f"{n.name.split('(')[0]} missing launch api, correlation id: {n.correlation_id}"
                        )
                    )

            logger.debug(
                (
                    "Some device events missing launch api: "
                    f"{', '.join([':'.join((k, str(v))) for k, v in node_count_dict.items()])}"
                )
            )

        staled_device_nodes = []
        for device_nodes in corrid_to_device.values():
            staled_device_nodes.extend(
                [n for n in device_nodes if n.type == EventTypes.KERNEL]
            )

        return tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list

    def _parse_node(
        self,
        event: DurationEvent,
        corrid_to_device: Dict[int, List[DeviceNode]],
        corrid_to_runtime: Dict[int, RuntimeNode],
        externalid_to_runtime: Dict[int, List[RuntimeNode]],
        tid2list: Dict[int, List[OperatorNode]],
        pl_tid2list: Dict[int, List[PLProfileNode]],
        tid2zero_rt_list: Dict[int, List[RuntimeNode]],
    ):
        corrid = event.correlation_id
        tid = event.tid
        if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]:
            device_node = DeviceNode.create(event)
            if corrid in corrid_to_runtime:
                rt_node = corrid_to_runtime[
                    corrid
                ]  # Don't pop it because it may be used by next kernel.
                if rt_node.device_nodes is None:
                    rt_node.device_nodes = []
                rt_node.device_nodes.append(device_node)

                # Check the external_id
                if rt_node.external_id != device_node.external_id:
                    logger.debug(
                        "Runtime and Device-op have same correlation id %s but with different external id!"
                        " (runtime external_id, device external_id): (%s, %s)"
                        % (corrid, rt_node.external_id, device_node.external_id)
                    )
            else:
                corrid_to_device[corrid].append(device_node)
        elif event.type == EventTypes.RUNTIME:
            device_nodes = corrid_to_device.pop(corrid, None)
            rt_node = RuntimeNode.create(event, device_nodes)
            corrid_to_runtime[corrid] = rt_node
            externalid_to_runtime[rt_node.external_id].append(rt_node)
            # Some runtimes has external_id 0, which will not be correlated to any operator.
            # So get them and attach them to root node.
            if rt_node.external_id == 0:
                tid2zero_rt_list[tid].append(rt_node)

            # check the external_id
            if device_nodes:
                for device_node in device_nodes:
                    if rt_node.external_id != device_node.external_id:
                        logger.debug(
                            "Runtime and Device-op have same correlation id %s but with different external id!"
                            " (rt external_id, device external_id): (%s, %s)"
                            % (corrid, rt_node.external_id, device_node.external_id)
                        )
        elif event.type in [
            EventTypes.PYTHON,
            EventTypes.OPERATOR,
            EventTypes.PL_MODULE,
            EventTypes.PROFILER_STEP,
            EventTypes.MODULE,
            EventTypes.USER_ANNOTATION,
        ]:
            if event.type == EventTypes.PROFILER_STEP:
                op_node = ProfilerStepNode.create(event)
            elif event.type == EventTypes.MODULE:
                op_node = ModuleNode.create(event)
            elif event.type == EventTypes.PL_MODULE:
                op_node = PLModuleNode.create(event)
            else:
                op_node = create_operator_node(event)
            if op_node:
                tid2list[int(tid)].append(op_node)
        elif event.type == EventTypes.PL_PROFILE:
            op_node = PLProfileNode.create(event)
            pl_tid2list[int(tid)].append(op_node)


class EventParser(NodeParserMixin):
    def __init__(self):
        super().__init__()

    def parse(self, events: Iterable[BaseEvent]) -> Dict[int, List[OperatorNode]]:
        tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list = self.parse_nodes(
            events
        )

        builder = OpTreeBuilder()
        tid2tree = builder.build_tree(tid2list, tid2zero_rt_list, staled_device_nodes)
        pl_tid2tree = builder.build_tree(pl_tid2list, {}, [])

        return tid2tree, pl_tid2tree

    @staticmethod
    def print_tree(root):
        class Ctx:
            tid: int = -1
            name_stack: list = []

        ctx = Ctx()

        def print_node_set_prefix(node: OperatorNode):
            header = f"[{ctx.tid}]" + ".".join(
                ctx.name_stack[1:]
            )  # omit the CallTreeRoot
            prefix_len = len(ctx.name_stack) * 4 - 4 - 1
            if len(ctx.name_stack) > 1:
                print(header)
                prefix = " " * prefix_len
                print(prefix, node.name)
                print(prefix, "time:", node.start_time, "-->", node.end_time)

        def push(node: OperatorNode):
            ctx.name_stack.append(node.name)

        def pop():
            ctx.name_stack.pop()

        def traverse_opeartor_node(node: OperatorNode):
            print_node_set_prefix(node)

            push(node)
            for n in node.children:
                traverse_opeartor_node(n)
            pop()

        ctx.tid = root.tid
        traverse_opeartor_node(root)
        ctx.tid = -1
