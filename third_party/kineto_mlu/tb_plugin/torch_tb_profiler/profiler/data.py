# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import gzip
import io as sysio
import json
import re
import tempfile
from json.decoder import JSONDecodeError
from typing import Dict, List, Optional

from .. import io, utils
from ..utils import href
from . import trace
from .communication import analyze_communication_nodes
from .event_parser import CommLibTypes, EventParser, ProfileRole
from .mlu_metrics_parser import MLUMetricsParser
from .kernel_parser import KernelParser
from .memory_parser import MemoryParser, MemorySnapshot
from .node import OperatorNode
from .op_agg import ModuleAggregator
from .overall_parser import OverallParser
from .tensor_cores_parser import TensorCoresParser
from .trace import BaseEvent, EventTypes, MemoryEvent

logger = utils.get_logger()


class RunProfileData(object):
    def __init__(self, worker: str, span: str, trace_json: Dict):
        self.worker = worker
        self.span = span

        # metadatas
        self.is_pytorch_lightning = trace_json.get('Framework', None) == 'pytorch-lightning'
        self.data_schema_version = trace_json.get('schemaVersion', None)
        self.distributed_info = trace_json.get('distributedInfo', None)
        self.device_props = trace_json.get('deviceProperties', None)

        self.profiler_start_ts = float('inf')
        self.events: List[BaseEvent] = []

        trace_body = trace_json['traceEvents']
        fwd_bwd_events = []
        for data in trace_body:
            if data.get('cat') == 'fwdbwd':
                fwd_bwd_events.append(data)
            else:
                event = trace.create_event(data, self.is_pytorch_lightning)
                if event is not None:
                    self.profiler_start_ts = min(self.profiler_start_ts, event.ts)
                    self.events.append(event)

        self.events.sort(key=lambda e: e.ts)
        self.forward_backward_events = trace.create_association_events(fwd_bwd_events)

        self.trace_file_path: str = None

        # Event Parser results
        self.tid2tree: Dict[int, OperatorNode] = None
        self.pl_tid2tree: Dict[int, OperatorNode] = None
        self.used_devices = []
        self.use_dp: bool = False
        self.use_ddp: bool = False
        self.comm_lib = None
        self.has_runtime: bool = False
        self.has_kernel: bool = False
        self.has_communication: bool = False
        self.has_memcpy_or_memset: bool = False
        self.role_ranges = None
        self.steps_costs = None
        self.steps_names = None
        self.avg_costs = None

        # MLU parser
        self.mlu_metrics_parser: MLUMetricsParser = None

        # Operator aggregator
        self.op_list_groupby_name = None
        self.op_list_groupby_name_input = None
        self.stack_lists_group_by_name = None
        self.stack_lists_group_by_name_input = None
        self.kernel_list_groupby_name_op = None

        # Kernel and Tensor Core
        self.kernel_stat = None
        self.tc_ratio = None
        self.tc_eligible_ops_kernel_ratio = None
        self.tc_used_ratio = None  # If it's a pure CPU run, then this keeps as None.

        # Communicator
        self.comm_node_list = None
        self.comm_overlap_costs = None
        self.memory_snapshot: Optional[MemorySnapshot] = None

        # recommendation based on analysis result.
        self.recommendations = []

    @staticmethod
    def parse(worker, span, path, cache_dir):
        trace_path, trace_json = RunProfileData._preprocess_file(path, cache_dir)

        profile = RunProfileData.from_json(worker, span, trace_json)
        profile.trace_file_path = trace_path
        return profile

    @staticmethod
    def from_json(worker, span, trace_json: Dict):
        profile = RunProfileData(worker, span, trace_json)
        with utils.timing('Data processing'):
            profile.process()
        profile.analyze()
        return profile

    @staticmethod
    def _preprocess_file(trace_path, cache_dir):
        if not io.exists(trace_path):
            raise FileNotFoundError(trace_path)

        data = io.read(trace_path)
        if trace_path.endswith('.gz'):
            data = gzip.decompress(data)

        json_reencode = False
        try:
            trace_json = json.loads(data)
        except JSONDecodeError as e:
            # Kineto may export json file with non-ascii code. before this is fixed, use a workaround
            # to handle JSONDecodeError, re-encode it and save to a temp file
            try:
                trace_json = json.loads(data, strict=False)
            except JSONDecodeError:
                with sysio.StringIO() as fout:
                    str_data = data.decode('utf-8')
                    # only replace the N/A without surrounding double quote
                    fout.write(re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data))
                    trace_json = json.loads(fout.getvalue())
                    logger.warning('Get JSONDecodeError: %s, Re-encode it to temp file' % e.msg)
                    json_reencode = True

        # work-around to remove the 'Record Window End' events to avoid the huge end timestamp
        event_list = trace_json['traceEvents']
        end_index = None
        start_index = None
        for i in reversed(range(len(event_list))):
            if event_list[i]['name'] == 'Record Window End':
                end_index = i
            elif event_list[i]['name'].startswith('Iteration Start:'):
                start_index = i
            if start_index is not None and end_index is not None:
                break

        if start_index is not None and end_index is not None:
            dur = event_list[end_index]['ts'] - event_list[start_index]['ts']
            if dur > 24 * 3600 * 1000:
                del trace_json['traceEvents'][end_index]
                json_reencode = True

        if json_reencode:
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', dir=cache_dir, delete=False)
            fp.close()
            with gzip.open(fp.name, mode='wt') as fzip:
                fzip.write(json.dumps(trace_json))
            trace_path = fp.name

        return trace_path, trace_json

    def process(self):
        with utils.timing('EventParser.parse'):
            parser = EventParser()
            self.tid2tree, self.pl_tid2tree = parser.parse(self.events, self.forward_backward_events)

        self.has_runtime = parser.has_runtime
        self.has_kernel = parser.has_kernel
        self.has_memcpy_or_memset = parser.has_memcpy_or_memset
        self.steps_names = parser.steps_names
        self.used_devices = sorted(list(parser.used_devices))
        self.use_dp = parser.use_dp
        self.use_ddp = parser.use_ddp
        self.role_ranges = parser.role_ranges

        self.comm_lib = parser.comm_lib
        self.has_communication = parser.has_communication
        self.comm_node_list = parser.comm_node_list

        # Starting aggregate
        logger.debug('ModuleAggregator')
        with utils.timing('ModuleAggregator aggegation'):
            module_aggregator = ModuleAggregator()
            module_aggregator.aggregate(self.tid2tree)
        self.op_list_groupby_name = module_aggregator.op_list_groupby_name
        self.op_list_groupby_name_input = module_aggregator.op_list_groupby_name_input
        self.stack_lists_group_by_name = module_aggregator.stack_lists_group_by_name
        self.stack_lists_group_by_name_input = module_aggregator.stack_lists_group_by_name_input
        self.kernel_list_groupby_name_op = module_aggregator.kernel_list_groupby_name_op

        logger.debug('OverallParser')
        with utils.timing('OverallParser aggegation'):
            overall_parser = OverallParser()
            overall_parser.aggregate(parser.steps, parser.role_ranges)
        self.avg_costs = overall_parser.avg_costs
        self.steps_costs = overall_parser.steps_costs
        self.comm_overlap_costs = overall_parser.communication_overlap

        logger.debug('MLUMetricsParser')
        self.mlu_metrics_parser = MLUMetricsParser.parse_events(
            self.events, parser.global_start_ts, parser.global_end_ts, parser.steps[0][0], parser.steps[-1][1])

        logger.debug('TensorCoresParser')
        tensorcores_parser = TensorCoresParser.parse_events(
            self.tid2tree, module_aggregator.ops, self.mlu_metrics_parser.mlu_ids)
        self.tc_eligible_ops_kernel_ratio = tensorcores_parser.tc_eligible_ops_kernel_ratio
        self.tc_ratio = tensorcores_parser.tc_ratio

        if self.has_kernel:
            logger.debug('KernelParser')
            with utils.timing('parse kernels'):
                kernel_parser = KernelParser()
                kernel_parser.parse_events(self.events)
            self.kernel_stat = kernel_parser.kernel_stat
            self.tc_used_ratio = kernel_parser.tc_used_ratio

        memory_events = self._memory_events()
        if memory_events:
            memory_parser = MemoryParser(memory_events)
            self.memory_snapshot = memory_parser.find_memory_nodes(self.tid2tree)

    def analyze(self):
        self.recommendations = []

        dataloader_ratio = self.avg_costs.costs[ProfileRole.DataLoader] / self.avg_costs.costs[ProfileRole.Total]
        if dataloader_ratio > 0.05:
            percentage = dataloader_ratio * 100
            url = 'https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading'
            self.recommendations.append(
                f'This run has high time cost on input data loading. {percentage:.1f}% of the step ' +
                "time is in DataLoader. You could try to set num_workers on DataLoader's construction " +
                f"and {href('enable multi-processes on data loading', url)}."
            )

        self._analyze_distributed_metrics()
        self._analyze_mlu_metrics()

        if self.device_props:
            # Memory related
            if self.memory_snapshot:
                for (dev_type, dev_id), peak_mem in self.memory_snapshot.get_peak_memory().items():
                    if dev_type == -1:  # ignore cpu
                        continue
                    total_mem = self.device_props[dev_id].get('totalMem(MiB)') * 1024 * 1024
                    if total_mem is not None and peak_mem > total_mem * 0.9:
                        percentage = peak_mem / total_mem * 100
                        total_mem_gb = total_mem / 1024 / 1024 / 1024
                        ckp_url = 'https://pytorch.org/docs/stable/checkpoint.html'
                        amp_url = 'https://pytorch.org/docs/stable/amp.html'
                        self.recommendations.append(
                            f'Device memory usage is at the limit of device memory capacity '
                            f'({percentage:.1f}% of {total_mem_gb:.1f}GB on MLU{dev_id}). '
                            'To get better value of your MLU or to use larger batch size for training, please refer to '
                            f"{href('Gradient Checkpoint', ckp_url)} or {href('Automatic Mixed Precision', amp_url)}.")
                        break

    def _analyze_distributed_metrics(self):
        if self.use_dp and len(self.used_devices) > 1:
            url = 'https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead'
            self.recommendations.append(
                f"It is recommended to {href('use DistributedDataParallel instead of DataParallel', url)}"
                ' to do multi-MLU training.')

        if self.use_ddp and CommLibTypes.Cncl not in self.comm_lib and self.device_props:
            text = (
                'Cncl backend is currently the fastest and highly recommended backend'
                ' when using DDP for training.')
            self.recommendations.append(text)

        communication_ratio = self.avg_costs.costs[ProfileRole.Communication] / self.avg_costs.costs[ProfileRole.Total]
        if communication_ratio > 0.1:
            percentage = communication_ratio * 100
            compress_url = 'https://pytorch.org/docs/stable/ddp_comm_hooks.html',
            grad_acc_url = 'https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa'
            lamb_url = 'https://nvidia.github.io/apex/optimizers.html#apex.optimizers.FusedLAMB'
            self.recommendations.append(
                f'This run has high time cost on communication. {percentage:.1f}% of the step time is in '
                f"communication. You could try {href('Gradient Compression', compress_url)} or "
                f"{href('Gradient Accumulation', grad_acc_url)} or increase the batch size. "
                'Note: Gradient accumulation will increase global effective batch size, which may hurt model '
                f"convergence and accuracy. For such case, you may want to evaluate {href('LAMB optimizer', lamb_url)}."
            )

    def _memory_events(self) -> List[MemoryEvent]:
        memory_events = [e for e in self.events if e.type == EventTypes.MEMORY]
        memory_events.sort(key=lambda e: e.ts)
        return memory_events

    def _analyze_mlu_metrics(self):
        def get_mlus_str(mlus):
            mlu_list_str = str(mlus[0])
            for i in range(1, len(mlus)):
                if i == len(mlus) - 1:
                    mlu_list_str += 'and {}'.format(mlus[i])
                else:
                    mlu_list_str += ', {}'.format(mlus[i])
            has_str = 'has' if len(mlu_list_str) == 1 else 'have'
            return mlu_list_str, has_str

        low_util_mlus = []
        for mlu_id in self.mlu_metrics_parser.mlu_ids:
            if self.mlu_metrics_parser.mlu_utilization[mlu_id] < 0.5:
                low_util_mlus.append(mlu_id)
        if len(low_util_mlus) > 0:
            mlu_list_str, has_str = get_mlus_str(low_util_mlus)
            text = 'MLU {} {} low utilization. You could try to ' \
                   'increase batch size to improve. Note: Increasing batch size ' \
                   'may affect the speed and stability of model convergence.'.format(mlu_list_str, has_str)
            self.recommendations.append(text)


class DistributedRunProfileData:
    def __init__(self, run_profile_data: RunProfileData):
        self.worker = run_profile_data.worker
        self.span = run_profile_data.span
        self.steps_names = run_profile_data.steps_names
        self.has_communication = run_profile_data.has_communication
        self.comm_lib = run_profile_data.comm_lib
        self.comm_node_list = run_profile_data.comm_node_list
        self.comm_overlap_costs = run_profile_data.comm_overlap_costs
        self.used_devices = run_profile_data.used_devices
        self.device_props = run_profile_data.device_props
        self.distributed_info = run_profile_data.distributed_info

        self.total_comm_stats = None
        self.step_comm_stats = None

    def communication_parse(self):
        self.step_comm_stats, self.total_comm_stats = analyze_communication_nodes(self.comm_node_list)
