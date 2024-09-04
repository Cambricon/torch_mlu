import os
import json
import gzip
import re
from json.decoder import JSONDecodeError
from typing import Dict, List, Optional

from . import trace
from .event_parser import EventParser
from .kernel_parser import KernelParser
from .memory_parser import MemoryParser, MemorySnapshot, MemoryRecord
from .node import OperatorNode, DeviceNode
from .trace import BaseEvent, EventTypes, MemoryEvent, DeviceType

from .common import utils
from .common import consts
from .common.file_manager import FileManager
from .common.path_manager import PathManager

logger = utils.get_logger()

WORKER_PATTERN = re.compile(
    r""".*/(.*?) # worker name
        \.(\d+)? # optional timestamp like 1619499959628 used as span name
        \.pt\.trace\.json # the ending suffix
        (?:\.gz)?$""",
    re.X,
)  # optional .gz extension


class ProfileData:
    def __init__(self, file_path: str):
        # metadatas
        self.raw_data_path = os.path.dirname(os.path.abspath(file_path))
        self.output_dir = self._create_output_dir(file_path)
        self.trace_json = self._load_file(file_path)

        self.events: List[BaseEvent] = []
        self.all_kernels: List[DeviceNode] = []

        if self.trace_json is not None:
            trace_body = self.trace_json["traceEvents"]
            fwd_bwd_events = []
            for data in trace_body:
                # discard fwdbwd flow event
                if (
                    data.get("cat") != "fwdbwd"
                    and data.get("cat") != "forward_backward"
                ):
                    event = trace.create_event(data)
                    if event is not None:
                        self.events.append(event)

            self.events.sort(key=lambda e: e.ts)

        # Event Parser results
        self.tid2tree: Dict[int, OperatorNode] = None
        self.pl_tid2tree: Dict[int, OperatorNode] = None
        self.memory_snapshot: Optional[MemorySnapshot] = None

    def _load_file(self, file_path: str):
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not exits, skip analysis.")

        with open(file_path, "rb") as f:
            data = f.read()
        if file_path.endswith(".gz"):
            data = gzip.decompress(data)

        try:
            trace_json = json.loads(data)
        except JSONDecodeError as e:
            logger.warning(
                f"JSONDecodeError: Failed to load json file {file_path}, skip analysis."
            )
            trace_json = None

        return trace_json

    def _create_output_dir(self, file_path: str):
        match = WORKER_PATTERN.match(file_path)
        worker = match.group(1) if match else "default_worker"
        span = match.group(2) if match else "default_span"
        output_dir = os.path.join(
            self.raw_data_path, consts.CAMBRICON_OUTPUT_DIR_NAME, f"{worker}-{span}"
        )
        PathManager.remove_path_safety(output_dir)
        PathManager.make_dir_safety(output_dir)
        return output_dir

    def process(self):
        parser = EventParser()
        self.tid2tree, self.pl_tid2tree = parser.parse(self.events)
        ops_details = []
        kerners_details = []
        l2_cache = []
        for _, root_node in self.tid2tree.items():
            ops, kernels = root_node.get_operator_and_kernels()
            self.all_kernels.extend(kernels)
            for op in ops:
                ops_details.append(op.data())
            for kernel in kernels:
                kerners_details.append(kernel.data())
                l2_cache.append(kernel.l2_cache_data())

        if ops_details:
            FileManager.create_csv_file(
                self.output_dir,
                ops_details,
                consts.OPERATOR_DETAILS_FILE_NAME,
                OperatorNode.header(),
            )
        if kerners_details:
            FileManager.create_csv_file(
                self.output_dir,
                kerners_details,
                consts.KERNEL_DETAILS_FILE_NAME,
                DeviceNode.header(),
            )
        if l2_cache:
            FileManager.create_csv_file(
                self.output_dir,
                l2_cache,
                consts.L2CACHE_FILE_NAME,
                DeviceNode.l2_cache_header(),
            )

        if self.all_kernels:
            kernel_parser = KernelParser()
            kernel_parser.parse_events(self.events, self.all_kernels)
            FileManager.create_csv_file(
                self.output_dir,
                kernel_parser.get_kernel_statistic(),
                consts.KERNEL_STATISTIC_FILE_NAME,
                kernel_parser.kernel_header,
            )

            FileManager.create_csv_file(
                self.output_dir,
                kernel_parser.get_op_kernel_statistic(),
                consts.OP_KERNEL_STATISTIC_FILE_NAME,
                kernel_parser.op_kernel_header,
            )

        memory_events = self._memory_events()
        if memory_events:
            memory_parser = MemoryParser(memory_events)
            self.memory_snapshot = memory_parser.find_memory_nodes(self.tid2tree)

            mlu_memory_records = []
            op_mem_events = []
            alloc = {}
            free = {}
            prev_ts = float("-inf")  # ensure ordered memory records is ordered
            OP_MEM_HEADER = [
                "Operator Name",
                "Size(KB)",
                "Allocation Time",
                "Release Time",
                "Duration(us)",
                "Address",
                "Device Type",
            ]
            for idx, record in enumerate(self.memory_snapshot.memory_records):
                # Only record MLU memory data
                if record.device_type is DeviceType.MLU:
                    mlu_memory_records.append(record.data())

                    # gen data for operator_memory.csv
                    assert prev_ts <= record.ts
                    prev_ts = record.ts
                    addr = record.addr
                    size = record.bytes
                    if record.is_allocation:
                        alloc[addr] = idx
                    else:
                        if addr in alloc:
                            alloc_record = self.memory_snapshot.memory_records[
                                alloc[addr]
                            ]
                            alloc_ts = alloc_record.ts
                            free_ts = record.ts
                            op_mem_events.append(
                                [
                                    alloc_record.full_op_name_or_unknow,  # op name
                                    -size / 1024.0,  # free record size is negtive
                                    alloc_ts,
                                    free_ts,
                                    free_ts - alloc_ts,
                                    addr,
                                    alloc_record.device_name,
                                ]
                            )
                            del alloc[addr]
                        else:
                            if addr in free:
                                logger.error(f"Address {addr} is freed multiple times")
                            free[addr] = idx

            for i in alloc.values():
                r = self.memory_snapshot.memory_records[i]
                op_mem_events.append(
                    [
                        r.full_op_name_or_unknow,  # op name
                        r.bytes / 1024.0,
                        r.ts,
                        None,
                        None,
                        r.addr,
                        r.device_name,
                    ]
                )

            for i in free.values():
                r = self.memory_snapshot.memory_records[i]
                op_mem_events.append(
                    [
                        r.full_op_name_or_unknow,  # op name
                        -r.bytes / 1024.0,
                        None,
                        r.ts,
                        None,
                        r.addr,
                        r.device_name,
                    ]
                )

            if mlu_memory_records:
                FileManager.create_csv_file(
                    self.output_dir,
                    mlu_memory_records,
                    consts.MEMORY_RECORD_FILE_NAME,
                    MemoryRecord.header(),
                )

            if op_mem_events:
                FileManager.create_csv_file(
                    self.output_dir,
                    op_mem_events,
                    consts.OPERATOR_MEMORY_FILE_NAME,
                    OP_MEM_HEADER,
                )

    def _memory_events(self) -> List[MemoryEvent]:
        memory_events = [e for e in self.events if e.type == EventTypes.MEMORY]
        memory_events.sort(key=lambda e: e.ts)
        return memory_events
