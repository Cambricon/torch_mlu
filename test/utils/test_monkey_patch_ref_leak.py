import os
import inspect
import warnings

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

import importlib
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

exclude_origin_list = [
    "torch.manual_seed",  # this will be patched when import _dynamo
]

exclude_other_list = [
    "torch._C",
    "torch.utils.data.dataset.Generator",
    "torch.serialization.load",
    "torch.cuda.random._lazy_call",
    "torch.distributed.distributed_c10d.batch_isend_irecv",
    "torch.distributed.init_process_group",
    "torch.distributed.new_group",
    "torch.autograd.profiler_util.EventList",
    "torch.autograd.profiler_legacy.EventList",
    "torch.autograd.profiler_util.FunctionEvent",
    "torch.autograd.profiler_legacy.FunctionEvent",
    "torch.jit._script.script",
    "torch.jit._trace.script",
    "torch.utils.data.dataset.randperm",
    "torch.obj",
]


def colored_text(text, color_code):
    return f"{color_code}{text}\033[0m"


def contains_any(substrings, string):
    return any(substring in string for substring in substrings)


class TestReferenceLeak:
    def test_ref_leak(self):
        modules = sys.modules

        attr_dict = {}
        for m in list(modules.keys()):
            if "torch" in m:
                module = modules[m]
                # check patched module
                attr_dict.setdefault(id(module), []).append([m])
                # check patched attr
                attrs = dir(module)
                for attr in attrs:
                    attr_obj = getattr(module, attr)
                    # skip some constant value
                    if (
                        attr_obj == None
                        or isinstance(attr_obj, (bool, int, float, complex, str))
                        or attr_obj in ((), {}, [])
                    ):
                        continue
                    attr_path = [m, attr]
                    attr_dict.setdefault(id(attr_obj), []).append(attr_path)

        import torch_mlu

        # Disable wrap warning functions
        # We don't check the unsupportted attr.
        os.environ["TORCH_MLU_MIGRATION_WITH_WARNING"] = "0"
        import torch_mlu.utils.gpu_migration

        # The list of suggested way of solving leak
        patch_list = []

        e_str = (
            colored_text(
                "\n" + " Referrence Leak When Monkey Patch ".center(100, "=") + "\n",
                "\033[31m",
            )
            + "\n"
        )
        e_str += (
            "The following prompt is written in the format: \n"
            + "origin object  ====> patched object \n"
            + "Leaks need to be patched \n"
            + colored_text("".center(100, "=") + "\n\n", "\033[31m")
        )
        e_flag = False
        for attr_id in attr_dict:
            for attr in attr_dict[attr_id]:
                attr_obj = None
                if len(attr) == 2:
                    mod = importlib.import_module(attr[0])
                    if hasattr(mod, attr[1]):
                        attr_obj = getattr(mod, attr[1])
                    else:
                        # This case is when a module is monkey patched, but some attr in origin
                        # module not exsit in the new one.
                        # We can skip this for that we have not monkey patched the origin attr indeed.
                        continue
                    attr_name = attr[0] + "." + attr[1]
                else:
                    attr_obj = importlib.import_module(attr[0])
                    attr_name = attr[0]
                # For some reasons, such as being patched by the original source, exclude certain attributes.
                flag = False
                for ex in exclude_origin_list:
                    if ex in attr_name:
                        flag = True
                        break
                if flag:
                    continue

                if id(attr_obj) != attr_id:
                    check_list = []
                    for other in attr_dict[attr_id]:
                        if len(other) == 2:
                            other_name = other[0] + "." + other[1]
                            other_mod = importlib.import_module(other[0])
                            # This is the case that the whole module was patched
                            # but there is origin attr of it not added, the reason may be that we don't support it.
                            # Here we just skip it.
                            if not hasattr(other_mod, other[1]):
                                continue
                            other_obj = getattr(
                                importlib.import_module(other[0]), other[1]
                            )
                        else:
                            other_obj = importlib.import_module(other[0])
                            other_name = other[0]
                        if (
                            # We assume that all obj should be patched to the same id
                            # If there are some special cases, please exclude it in exclude_other_list
                            id(other_obj) != id(attr_obj)
                            and not contains_any(exclude_other_list, other_name)
                            and other_name != attr_name
                        ):
                            check_list.append(other_name)
                    if len(check_list) > 0:
                        e_flag = True
                        if inspect.ismodule(attr_obj):
                            patched_obj_name = attr_obj.__name__
                        else:
                            patched_obj_name = (
                                str(attr_obj.__module__) + "." + attr_obj.__name__
                            )
                        e_str += attr_name + " ====> " + patched_obj_name + "\n"
                        e_str += "Leaks To Patch: \n"
                        for to_patch in check_list:
                            e_str += "                " + to_patch + "\n"
                            origin = attr[0] + "." + attr[1]
                            if to_patch == patched_obj_name:
                                patch_list.append(to_patch + " = " + origin + "\n")
                            elif attr == to_patch:
                                patch_list.append(
                                    "sys.modules["
                                    + origin
                                    + " = "
                                    + patched_obj_name
                                    + "\n"
                                )
                            else:
                                patch_list.append(
                                    to_patch + " = " + patched_obj_name + "\n"
                                )

                        e_str += "\n"
                    # Once an attr was detected monkey patched, all other attr in it's list need not to be checked agained
                    break
        if e_flag:
            e_str += (
                colored_text(
                    " Suggesting Ways To Fix These Leaks ".center(100, "="), "\033[31m"
                )
                + "\n"
            )
            e_str += (
                colored_text(
                    " The following methods can not guarantee to be correct, PLEASE MAKE DOUBLE CHECK!!! ".center(
                        100, "="
                    ),
                    "\033[31m",
                )
                + "\n"
            )
            for to_patch in patch_list:
                e_str += str(to_patch) + "\n"

            e_str += colored_text("".center(100, "="), "\033[31m") + "\n"
            raise ValueError(e_str)
