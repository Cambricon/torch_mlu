import os
import warnings
from torch_mlu._MLUC import _dump_cnnl_gencase as dump_cnnl_gencase

_GENCASE_ENABLED = False
_GENCASE_STATUS_DICT = {
    "cur_count": 0,
    "start_count": 0,
    "end_count": 0,
    "gencase_level": "L1",
}

_ENV_TO_GENCASE_LEVEL = {"L1": 1, "L2": 2, "L3": 3}


def _check_gencase():
    global _GENCASE_ENABLED, _GENCASE_STATUS_DICT
    start_count = os.environ.get("TORCH_MLU_COUNTER_START")
    end_count = os.environ.get("TORCH_MLU_COUNTER_END")
    gencase_level = os.environ.get("TORCH_MLU_COUNTER_LEVEL")
    gencase_level_list = ["L1", "L2", "L3"]

    if start_count is not None and end_count is not None and gencase_level is not None:
        if gencase_level not in gencase_level_list:
            warnings.warn(
                f"TORCH_MLU_COUNTER enable failed, TORCH_MLU_COUNTER_LEVEL only support "
                f"L1, L2, L3, but got {gencase_level}"
            )
            return

        try:
            start_count = int(start_count)
            end_count = int(end_count)
        except ValueError:
            warnings.warn(
                "TORCH_MLU_COUNTER_START and TORCH_MLU_COUNTER_END should be integers"
            )
            return

        if start_count < 0 or end_count < 0:
            warnings.warn(
                "TORCH_MLU_COUNTER_START and TORCH_MLU_COUNTER_END should all be greater than 0"
            )
            return

        if end_count > start_count:
            _GENCASE_ENABLED = True
            _GENCASE_STATUS_DICT["start_count"] = start_count
            _GENCASE_STATUS_DICT["end_count"] = end_count
            _GENCASE_STATUS_DICT["gencase_level"] = gencase_level
        else:
            warnings.warn(
                "TORCH_MLU_COUNTER_END should be greater than TORCH_MLU_COUNTER_START"
            )
            return
    elif start_count is not None or end_count is not None or gencase_level is not None:
        warnings.warn(
            "TORCH_MLU_COUNTER_START, TORCH_MLU_COUNTER_END, TORCH_MLU_COUNTER_LEVEL should be"
            " all set to enable torch_mlu counter"
        )
        return


def _update_and_check_for_gencase():
    global _GENCASE_ENABLED, _GENCASE_STATUS_DICT
    if not _GENCASE_ENABLED:
        return
    cur_count = _GENCASE_STATUS_DICT["cur_count"]
    if cur_count == _GENCASE_STATUS_DICT["start_count"]:
        dump_cnnl_gencase(_ENV_TO_GENCASE_LEVEL[_GENCASE_STATUS_DICT["gencase_level"]])

    elif cur_count >= _GENCASE_STATUS_DICT["end_count"]:
        print(f"final TORCH_MLU_COUNTER is {cur_count - 1}")
        _GENCASE_ENABLED = False
        dump_cnnl_gencase(0)
        return

    cur_count = cur_count + 1
    _GENCASE_STATUS_DICT["cur_count"] = cur_count
    print(f"current TORCH_MLU_COUNTER is {cur_count - 1}")
    return
