import warnings
import os
import torch_mlu


def dump_cnnl_gencase(enable=True, level="L1"):
    r"""
    Dump imformations about CNNL Kernels.

    Arguments:
        enable (bool): Turn on/off the cnnl gencase API.
        level (str): Three mode that cnnl API provide,
            'L1' : dump shapes, dtype, layout and args of CNNL kernels to files.
            'L2' : dump shapes, dtype, layout and args of CNNL Kernels to files,
                   export CNNL_GEN_CASE_DUMP_DATA=1 if you want to save real input data.
            'L3' : print shapes, dtype layout and args of CNNL Kernels to screen.

    Return :
         The cnnl gencase API state(on/off).
    """
    cnnl_gencase_env = os.environ.get("CNNL_GEN_CASE")
    if cnnl_gencase_env is not None:
        warnings.warn(
            "Please unset enviroment variable 'CNNL_GEN_CASE',"
            "or the dump_cnnl_gencase method will not work."
        )
        return False

    _level = ["L1", "L2", "L3"]
    if not enable:
        torch_mlu._MLUC._dump_cnnl_gencase(0)
        return False
    if level not in _level:
        msg = (
            f"dump_cnnl_gencase only support level"
            f"'L1' 'L2' and 'L3', but get level {level}."
        )
        raise ValueError(msg)

    if level == _level[0]:
        torch_mlu._MLUC._dump_cnnl_gencase(1)
    elif level == _level[1]:
        dump_data_env = os.environ.get("CNNL_GEN_CASE_DUMP_DATA")
        if not dump_data_env:
            warnings.warn(
                "Please export CNNL_GEN_CASE_DUMP_DATA=1,"
                "if you want to save real input data."
            )
        torch_mlu._MLUC._dump_cnnl_gencase(2)
    else:
        torch_mlu._MLUC._dump_cnnl_gencase(3)
    return True
