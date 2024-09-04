from typing import List, Tuple

import logging
import os
import re

def get_logging_level():
    log_level = os.environ.get("TORCH_PROFILER_LOG_LEVEL", "INFO").upper()
    if log_level not in logging._levelToName.values():
        log_level = logging.getLevelName(logging.INFO)
    return log_level


logger = None


def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger("MLU Profiler Analysis")
        logger.setLevel(get_logging_level())
    return logger


def merge_ranges(src_ranges, is_sorted=False) -> List[Tuple[float, float]]:
    if not src_ranges:
        # return empty list if src_ranges is None or its length is zero.
        return []

    if not is_sorted:
        src_ranges.sort(key=lambda x: x[0])

    merged_ranges = []
    merged_ranges.append(src_ranges[0])
    for src_id in range(1, len(src_ranges)):
        src_range = src_ranges[src_id]
        if src_range[1] > merged_ranges[-1][1]:
            if src_range[0] <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], src_range[1])
            else:
                merged_ranges.append((src_range[0], src_range[1]))

    return merged_ranges

def reduce_name(name: str) -> str:
    RETURN_PATTERN = re.compile(r'(?:void\s+)?([^\(<]+)')

    matched = RETURN_PATTERN.match(name)
    if matched:
        name = matched.group(1)

    return name
