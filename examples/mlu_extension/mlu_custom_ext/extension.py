import os
import sys
import importlib.machinery
import torch


def _get_extension_path(lib_name):
    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES,
    )
    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError
    return ext_specs.origin


_HAS_OPS = False
try:
    lib_path = _get_extension_path("_C")
    torch.ops.load_library(lib_path)
    _HAS_OPS = True
except (ImportError, OSError):
    print("not find lib")
    pass
