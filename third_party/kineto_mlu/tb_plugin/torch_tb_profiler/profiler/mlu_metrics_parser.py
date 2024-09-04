# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from typing import Iterable, List

from .. import consts, utils
from .range_utils import (get_ranges_sum, intersection_ranges_lists,
                          intersection_ranges_lists_with_value, merge_ranges,
                          merge_ranges_with_value)
from .trace import BaseEvent, EventTypes, KernelEvent

logger = utils.get_logger()


# For calculating MLU utilization, and approximated SM efficiency.
class MLUMetricsParser(object):
    def __init__(self):
        # All mlu ids that used by any kernel.
        self.mlu_ids = set()
        # For calculating MLU utilization.
        self.kernel_ranges_per_device = [[] for _ in range(consts.MAX_MLU_PER_NODE)]
        self.mlu_utilization = [None] * consts.MAX_MLU_PER_NODE
        self.mlu_util_timeline_unit_size = 0
        self.mlu_util_timeline_unit_name = ''
        self.mlu_util_buckets = [[] for _ in range(consts.MAX_MLU_PER_NODE)]
        # For calculating approximated SM efficiency.
        self.blocks_per_sm_per_device = [[] for _ in range(consts.MAX_MLU_PER_NODE)]
        self.avg_approximated_sm_efficiency_per_device = [None] * consts.MAX_MLU_PER_NODE
        self.approximated_sm_efficiency_ranges = [[] for _ in range(consts.MAX_MLU_PER_NODE)]
        self.mlu_sm_efficiency_json = None
        self.blocks_per_sm_count = [0] * consts.MAX_MLU_PER_NODE
        # For calculating averaged occupancy.
        self.occupancy_per_device = [[] for _ in range(consts.MAX_MLU_PER_NODE)]
        self.avg_occupancy_per_device = [None] * consts.MAX_MLU_PER_NODE
        self.occupancy_count = [0] * consts.MAX_MLU_PER_NODE

    def calculate_mlu_utilization(self, global_start_time, global_end_time, steps_start_time, steps_end_time):
        # Make bucket_size to 10-power's of us, and number of buckets to (10, 100].
        # 10-power's of us, in order to straight forward for user to understand.
        # If number of buckets are too many, the value of mlu utilization will be either 0 or 1.
        def get_bucket_info(range_micro_seconds):
            max_buckets = 100
            bucket_size = 1
            while range_micro_seconds / bucket_size > max_buckets:
                bucket_size *= 10
            buckets = int(range_micro_seconds / bucket_size)
            unit = bucket_size
            unit_str = 'us'
            if unit >= 1000:
                unit /= 1000
                unit_str = 'ms'
                if unit >= 1000:
                    unit /= 1000
                    unit_str = 's'
            return int(bucket_size), int(buckets), int(unit), unit_str

        mlu_utilization_timeline = [[] for _ in range(consts.MAX_MLU_PER_NODE)]
        for mlu_id in self.mlu_ids:
            self.kernel_ranges_per_device[mlu_id] = merge_ranges(self.kernel_ranges_per_device[mlu_id])

            # Top-level number still consider steps, to be consistent with overview's breakdown.
            kernel_ranges_all_steps = intersection_ranges_lists(
                self.kernel_ranges_per_device[mlu_id], [(steps_start_time, steps_end_time)])
            ranges_sum = get_ranges_sum(kernel_ranges_all_steps)
            self.mlu_utilization[mlu_id] = ranges_sum / (steps_end_time - steps_start_time)

            # The timeline will use 'PyTorch Profiler (0)' as start,
            # in order to draw previous step's kernels' mlu utilization.
            bucket_size, buckets, self.mlu_util_timeline_unit_size, self.mlu_util_timeline_unit_name = \
                get_bucket_info(global_end_time - global_start_time)
            buckets_ranges = []
            for i in range(buckets):
                buckets_ranges.append((global_start_time + i * bucket_size,
                                       global_start_time + (i + 1) * bucket_size if i < buckets - 1
                                       else global_end_time))  # The last bucket may be longer.
            mlu_utilization_timeline[mlu_id] = [0] * buckets
            if len(self.kernel_ranges_per_device[mlu_id]) > 0:
                current_range_index = 0
                current_range = self.kernel_ranges_per_device[mlu_id][current_range_index]
                current_bucket_index = 0
                current_bucket = buckets_ranges[0]
                while (current_range_index < len(self.kernel_ranges_per_device[mlu_id])
                       and current_bucket_index < buckets):
                    if current_bucket[1] <= current_range[0]:
                        current_bucket_index += 1
                        current_bucket = buckets_ranges[current_bucket_index] if current_bucket_index < buckets \
                            else None
                    elif current_bucket[0] >= current_range[1]:
                        current_range_index += 1
                        if current_range_index < len(self.kernel_ranges_per_device[mlu_id]):
                            current_range = self.kernel_ranges_per_device[mlu_id][current_range_index]
                    else:
                        left_bound = max(current_range[0], current_bucket[0])
                        right_bound = min(current_range[1], current_bucket[1])
                        mlu_utilization_timeline[mlu_id][current_bucket_index] += (right_bound - left_bound)
                        if current_bucket[1] < current_range[1]:
                            current_bucket_index += 1
                            current_bucket = buckets_ranges[current_bucket_index] if current_bucket_index < buckets \
                                else None
                        else:
                            current_range_index += 1
                            if current_range_index < len(self.kernel_ranges_per_device[mlu_id]):
                                current_range = self.kernel_ranges_per_device[mlu_id][current_range_index]
                for i_bucket in range(buckets):
                    bucket_size = buckets_ranges[i_bucket][1] - buckets_ranges[i_bucket][0]
                    mlu_utilization_timeline[mlu_id][i_bucket] /= bucket_size
                    start_time = buckets_ranges[i_bucket][0]
                    self.mlu_util_buckets[mlu_id].append((start_time, mlu_utilization_timeline[mlu_id][i_bucket]))
                start_time = buckets_ranges[-1][1]
                self.mlu_util_buckets[mlu_id].append((start_time, 0))

        self.kernel_ranges_per_device = None  # Release memory.

    def calculate_approximated_sm_efficiency(self, steps_start_time, steps_end_time):
        def calculate_avg(approximated_sm_efficiency_ranges, total_dur):
            total_weighted_sm_efficiency = 0.0
            for r in approximated_sm_efficiency_ranges:
                dur = r[1] - r[0]
                total_weighted_sm_efficiency += r[2] * dur
            avg_approximated_sm_efficiency = total_weighted_sm_efficiency / total_dur
            return avg_approximated_sm_efficiency

        total_dur = steps_end_time - steps_start_time
        for mlu_id in self.mlu_ids:
            blocks_per_sm_ranges = self.blocks_per_sm_per_device[mlu_id]
            approximated_sm_efficiency_ranges = merge_ranges_with_value(blocks_per_sm_ranges)
            # To be consistent with MLU utilization, here it must also intersect with all steps,
            # in order to remove the kernels out of steps range.
            approximated_sm_efficiency_ranges_all_steps = intersection_ranges_lists_with_value(
                approximated_sm_efficiency_ranges, [(steps_start_time, steps_end_time)])
            if len(approximated_sm_efficiency_ranges_all_steps) > 0:
                avg_approximated_sm_efficiency = calculate_avg(approximated_sm_efficiency_ranges_all_steps, total_dur)
                self.avg_approximated_sm_efficiency_per_device[mlu_id] = avg_approximated_sm_efficiency

            # The timeline still uses all kernels including out of steps scope's.
            if len(approximated_sm_efficiency_ranges) > 0:
                self.approximated_sm_efficiency_ranges[mlu_id] = approximated_sm_efficiency_ranges

        self.blocks_per_sm_per_device = None  # Release memory.

    # Weighted average. Weighted by kernel's time duration.
    def calculate_occupancy(self, steps_start_time, steps_end_time):
        for mlu_id in self.mlu_ids:
            occupancys_on_a_device = self.occupancy_per_device[mlu_id]
            total_time = 0
            total_occupancy = 0.0
            for r in occupancys_on_a_device:
                min_time = max(r[0], steps_start_time)
                max_time = min(r[1], steps_end_time)
                if min_time < max_time:
                    dur = max_time - min_time
                    total_occupancy += r[2] * dur
                    total_time += dur
            if total_time > 0:
                self.avg_occupancy_per_device[mlu_id] = total_occupancy / total_time

    @classmethod
    def parse_events(cls,
                     events: Iterable[BaseEvent],
                     global_start_time: int,
                     global_end_time: int,
                     steps_start_time: int,
                     steps_end_time: int):
        parser = MLUMetricsParser()
        logger.debug('MLU Metrics, parse events')
        for event in events:
            if event.type == EventTypes.KERNEL:
                parser.parse_event(event)

        parser.calculate_mlu_utilization(global_start_time, global_end_time, steps_start_time, steps_end_time)
        parser.calculate_approximated_sm_efficiency(steps_start_time, steps_end_time)
        parser.calculate_occupancy(steps_start_time, steps_end_time)
        return parser

    def parse_event(self, event: KernelEvent):
        ts = event.ts
        dur = event.duration
        mlu_id = event.device_id
        if mlu_id != event.pid:
            logger.warning("pid '{}' is not equal to args.device '{}' on event with ts '{}'".format(
                event.pid, mlu_id, event.ts))
        if mlu_id is not None:
            if mlu_id not in self.mlu_ids:
                self.mlu_ids.add(mlu_id)
            self.kernel_ranges_per_device[mlu_id].append((ts, ts + dur))
            if event.blocks_per_sm is not None:
                if event.blocks_per_sm > 0.0:
                    self.blocks_per_sm_per_device[mlu_id].append((ts, ts + dur, event.blocks_per_sm))
                    self.blocks_per_sm_count[mlu_id] += 1
                else:
                    # Workaround for negative value input.
                    logger.warning('blocks per SM {} with ts {} is not positive!'.format(event.blocks_per_sm, ts))
            if event.occupancy is not None:
                if event.occupancy >= 0.0:
                    self.occupancy_per_device[mlu_id].append((ts, ts + dur, event.occupancy))
                    self.occupancy_count[mlu_id] += 1
                else:
                    # Workaround for negative value input.
                    logger.warning('est. achieved occupancy % {} with ts {} is negative!'.format(event.occupancy, ts))

    def get_mlu_metrics_columns(self):
        columns = []
        if self.has_blocks_per_sm:
            columns.append({'type': 'number', 'name': 'Mean Blocks Per SM',
                            'tooltip': consts.TOOLTIP_BLOCKS_PER_SM})
        if self.has_occupancy:
            columns.append({'type': 'number', 'name': 'Mean Est. Achieved Occupancy (%)',
                            'tooltip': consts.TOOLTIP_OCCUPANCY_COMMON + consts.TOOLTIP_OCCUPANCY_TABLE})
        return columns

    @property
    def has_blocks_per_sm(self):
        return sum(self.blocks_per_sm_count) > 0

    @property
    def has_occupancy(self):
        return sum(self.occupancy_count) > 0

    def get_mlu_metrics(self):
        def build_trace_counter_mlu_util(mlu_id, start_time, counter_value):
            util_json = ("{{\"ph\":\"C\", \"name\":\"MLU {} Utilization\", \"pid\":{}, \"ts\":{}, "
                         "\"args\":{{\"MLU Utilization\":{}}}}}").format(mlu_id, mlu_id, start_time, counter_value)
            return util_json

        def add_trace_counter_mlu_util(mlu_id, start_time, counter_value, counter_json_list: List):
            json_str = build_trace_counter_mlu_util(mlu_id, start_time, counter_value)
            counter_json_list.append(json_str)

        counter_json_list = []
        for mlu_id, buckets in enumerate(self.mlu_util_buckets):
            if len(buckets) > 0:
                # Adding 1 as baseline. To avoid misleading virtualization when the max value is less than 1.
                add_trace_counter_mlu_util(mlu_id, buckets[0][0], 1, counter_json_list)
                add_trace_counter_mlu_util(mlu_id, buckets[0][0], 0, counter_json_list)
            for b in buckets:
                add_trace_counter_mlu_util(mlu_id, b[0], b[1], counter_json_list)

        return counter_json_list

    def get_mlu_metrics_data_tooltip(
            self,
            mlu_infos,
            tc_ratio):
        if not self.mlu_ids:
            return None, None

        has_sm_efficiency = False
        has_occupancy = False
        has_tc = False

        mlu_metrics_data = []
        mlu_info_columns = ['Name', 'Memory', 'Compute Capability']

        def process_mlu(mlu_id: int):
            nonlocal has_sm_efficiency, has_occupancy, has_tc
            mlu_metrics_data.append({'title': 'MLU {}:'.format(mlu_id), 'value': ''})
            mlu_info = mlu_infos.get(mlu_id, None)
            if mlu_info is not None:
                for key in mlu_info_columns:
                    if key in mlu_info:
                        mlu_metrics_data.append({'title': key, 'value': mlu_info[key]})
            else:
                # the legacy chrome tracing file would not have mlu info.
                pass
            mlu_metrics_data.append({'title': 'MLU Utilization', 'value': '{} %'.format(
                round(self.mlu_utilization[mlu_id] * 100, 2))})
            if self.avg_approximated_sm_efficiency_per_device[mlu_id] is not None:
                mlu_metrics_data.append({'title': 'Est. SM Efficiency', 'value': '{} %'.format(
                    round(self.avg_approximated_sm_efficiency_per_device[mlu_id] * 100, 2))})
                has_sm_efficiency = True
            if self.avg_occupancy_per_device[mlu_id] is not None:
                mlu_metrics_data.append({'title': 'Est. Achieved Occupancy', 'value': '{} %'.format(
                    round(self.avg_occupancy_per_device[mlu_id], 2))})
                has_occupancy = True
            if tc_ratio[mlu_id] is not None:
                mlu_metrics_data.append({'title': 'Kernel Time using Tensor Cores', 'value': '{} %'.format(
                    round(tc_ratio[mlu_id] * 100, 2))})
                has_tc = True

        mlu_ids = list(self.mlu_ids)
        process_mlu(mlu_ids[0])
        for idx in range(1, len(mlu_ids)):
            # Append separator line for beautiful to see.
            mlu_metrics_data.append({'title': '<hr/>', 'value': ''})
            process_mlu(mlu_ids[idx])

        tooltip_summary = 'The MLU usage metrics:\n'
        tooltip = '{}\n{}'.format(tooltip_summary,  consts.TOOLTIP_MLU_UTIL)
        if has_sm_efficiency:
            tooltip += '\n' + consts.TOOLTIP_SM_EFFICIENCY
        if has_occupancy:
            tooltip += '\n' + consts.TOOLTIP_OCCUPANCY_COMMON + consts.TOOLTIP_OCCUPANCY_OVERVIEW
        if has_tc:
            tooltip += '\n' + consts.TOOLTIP_TENSOR_CORES

        return mlu_metrics_data, tooltip
