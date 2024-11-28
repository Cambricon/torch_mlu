# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .common import utils
from .node import (
    BackwardNode,
    DeviceNode,
    ModuleNode,
    OperatorNode,
    UserAnnotationNode,
    ProfilerStepNode,
    RuntimeNode,
    is_operator_node,
)
from .trace import EventTypes

logger = utils.get_logger()


class OpTreeBuilder:
    def __init__(self):
        self.tid2tree: Dict[int, OperatorNode] = None
        self.external_id_to_anno: Dict[int, UserAnnotationNode] = {}

    def _set_annotation_parent(self, tail_node, cncl_anno_list):
        for cncl_anno in cncl_anno_list:
            if (
                tail_node.name != "CallTreeRoot"
                and cncl_anno.start_time > tail_node.start_time
                and cncl_anno.end_time <= tail_node.end_time
                and cncl_anno.external_id not in self.external_id_to_anno.keys()
            ):
                cncl_anno.set_parent(tail_node)
                self.external_id_to_anno[cncl_anno.external_id] = cncl_anno
                break

    def build_tree(
        self,
        tid2list: Dict[int, List[OperatorNode]],
        tid2zero_rt_list: Dict[int, List[RuntimeNode]],
        staled_device_nodes: List[DeviceNode],
        tid2cncl_annos: Dict[int, List[UserAnnotationNode]],
        id2opinfo: Dict = {},
    ):
        self.tid2tree = self._build_tree(
            tid2list, tid2zero_rt_list, staled_device_nodes, tid2cncl_annos, id2opinfo
        )
        return self.tid2tree

    def _build_tree(
        self,
        tid2list: Dict[int, List[OperatorNode]],
        tid2zero_rt_list,
        staled_device_nodes,
        tid2cncl_annos,
        id2opinfo: Dict = {},
    ):
        tid2tree = {}

        for tid, op_list in tid2list.items():
            zero_rt_list = tid2zero_rt_list[tid] if tid in tid2zero_rt_list else []
            cncl_anno_list = tid2cncl_annos[tid] if tid in tid2cncl_annos else []
            cncl_anno_list.sort(key=lambda x: (x.start_time, -x.end_time))
            # Note that when 2 start_time are equal, the one with bigger end_time should be ahead of the other.
            op_list.sort(key=lambda x: (x.start_time, -x.end_time))
            root_node = self._build_tree_internal(
                op_list, zero_rt_list, tid, [], cncl_anno_list, id2opinfo
            )
            tid2tree[int(tid)] = root_node

        # Some threads not created by PyTorch, link their runtime events to a new
        # CallTreeRoot to gather their kernels.
        for tid, op_list in tid2zero_rt_list.items():
            if tid not in tid2tree:
                tid2tree[int(tid)] = self._build_tree_internal(
                    [], op_list, tid, [], id2opinfo
                )

        # The staled_device_nodes contain kernels that cannot find the launch api, usually the kernel of the previous step.
        # Set condition always to False to skip saving staled device nodes to csv file. They are not launched by current step.
        # Keep this code in case we need this data in the future.
        if False and staled_device_nodes:
            # Add staled device nodes to a new CallTreeRoot.
            tid2tree[0] = self._build_tree_internal([], [], 0, staled_device_nodes, [])

        return tid2tree

    def _build_tree_internal(
        self,
        host_node_list,
        zero_rt_list,
        tid,
        staled_device_nodes,
        cncl_anno_list,
        id2opinfo: Dict = {},
    ):
        """host_node_list: list of OperatorNode and ProfilerStepNode.
        zero_rt_list: list of RuntimeNode with external_id=0."""

        def build_tree_relationship(
            host_node_list: Iterable[OperatorNode],
            zero_rt_list,
            staled_device_nodes,
            id2opinfo: Dict = {},
        ):
            dummpy_rt: List[RuntimeNode] = []
            if staled_device_nodes:
                # Note: Although kernels of this dummy runtime is put under main thread's tree,
                # we don't know which thread launches them.
                # TODO: Don't make belonging thread assumption on future usage if we need special handling
                dummpy_rt.append(
                    RuntimeNode(
                        name="dummy",
                        start_time=None,
                        end_time=None,
                        type=EventTypes.RUNTIME,
                        tid=0,
                        device_nodes=staled_device_nodes,
                    )
                )
                dummpy_rt[0].fill_stats(id2opinfo)

            # We build tree relationship also for runtime nodes with external_id=0, in order to
            # show kernel's op_node name by the topmost rt_node name rather than 'CallTreeRoot'.
            zero_rt_list.sort(key=lambda x: (x.start_time, -x.end_time))
            rt_stack: List[RuntimeNode] = []
            root_zero_rt = RuntimeNode(
                name="DummyZeroRuntimeRoot",
                start_time=-sys.maxsize - 1,
                end_time=sys.maxsize,
                type=EventTypes.PYTHON,
                tid=tid,
                parent_rt_node=None,
            )
            rt_stack.append(root_zero_rt)
            for node in zero_rt_list:
                while True:  # break loop when the node is inserted.
                    tail_node = rt_stack[-1]
                    if node.start_time < tail_node.end_time:
                        if node.end_time <= tail_node.end_time:
                            node.parent_rt_node = tail_node
                            rt_stack.append(node)
                        else:
                            logger.error(
                                "Error in input data: ranges on the same thread should not intersect!"
                                "Father:({},{},{}) Child:({},{},{})".format(
                                    tail_node.name,
                                    tail_node.start_time,
                                    tail_node.end_time,
                                    node.name,
                                    node.start_time,
                                    node.end_time,
                                )
                            )
                        break
                    else:
                        rt_stack.pop()

            node_stack: List[OperatorNode] = []
            root_node = OperatorNode(
                name="CallTreeRoot",
                start_time=-sys.maxsize - 1,
                end_time=sys.maxsize,
                type=EventTypes.PYTHON,
                tid=tid,
                runtimes=zero_rt_list + dummpy_rt,
            )  # Give the list of RuntimeNode with external_id=0 to root node.
            node_stack.append(root_node)
            for node in host_node_list:
                while True:  # break loop when the node is inserted.
                    tail_node = node_stack[-1]
                    if node.start_time < tail_node.end_time:
                        if node.end_time <= tail_node.end_time:
                            tail_node.children.append(node)
                            # node.parent_node = weakref.ref(tail_node)
                            node_stack.append(node)
                        else:
                            logger.error(
                                "Error in input data: ranges on the same thread should not intersect!"
                                "Father:({},{},{}) Child:({},{},{})".format(
                                    tail_node.name,
                                    tail_node.start_time,
                                    tail_node.end_time,
                                    node.name,
                                    node.start_time,
                                    node.end_time,
                                )
                            )
                        break
                    else:
                        # cncl user_annotation at the top of op node stack.
                        self._set_annotation_parent(tail_node, cncl_anno_list)
                        node_stack.pop()
            # For the scenario where the last node_stack contain a communication operator.
            tail_node = node_stack[-1]
            while tail_node != root_node:
                self._set_annotation_parent(tail_node, cncl_anno_list)
                node_stack.pop()
                tail_node = node_stack[-1]
            return root_node

        # Merge the consecutive calls to same function into one.
        # Just follow the same pattern in torch/autograd/profiler.py,
        # EventList._remove_dup_nodes
        # TODO: Replace recursive by for loop, in case of too deep callstack.
        def remove_dup_nodes(node: OperatorNode):
            if node.type == EventTypes.RUNTIME:
                return
            if len(node.children) == 1:
                child = node.children[0]
                if (
                    node.name == child.name
                    and node.type == EventTypes.OPERATOR
                    and child.type == EventTypes.OPERATOR
                ):
                    node.children = child.children
                    node.runtimes = (
                        child.runtimes
                    )  # Keep consistent with autograd profiler.
                    remove_dup_nodes(
                        node
                    )  # This node may have to merge with child's child.
                    return

            for child in node.children:
                remove_dup_nodes(child)

        root_node = build_tree_relationship(
            host_node_list, zero_rt_list, staled_device_nodes, id2opinfo
        )
        remove_dup_nodes(root_node)
        root_node.fill_stats(id2opinfo)

        # replace the root_node start_time/end_time
        root_node.start_time = next(
            (
                child.start_time
                for child in root_node.children
                if child.start_time is not None
            ),
            None,
        )
        root_node.end_time = next(
            (
                child.end_time
                for child in reversed(root_node.children)
                if child.end_time is not None
            ),
            None,
        )
        return root_node
