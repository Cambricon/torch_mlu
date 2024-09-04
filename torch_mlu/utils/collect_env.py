import datetime
import re
import sys
import os
from collections import namedtuple

import torch
from torch.utils import collect_env as torch_collect_env


try:
    import torch_mlu

    TORCH_MLU_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_MLU_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "torch_mlu_version",
        "is_debug_build",
        "os",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "libc_version",
        "python_version",
        "python_platform",
        "caching_allocator_config",
        "is_mlu_available",
        "is_xnnpack_available",
        "mlu_driver_version",
        "cnrt_runtime_version",
        "cndev_runtime_version",
        "cndrv_runtime_version",
        "cnpapi_runtime_version",
        "cndali_runtime_version",
        "cnnl_runtime_verson",
        "mluops_runtime_version",
        "cncl_runtime_version",
        "cncv_runtime_version",
        "cnnlextra_runtime_version",
        "cpu_info",
        "pip_version",
        "pip_packages",
        "conda_packages",
    ],
)


def get_cachingallocator_config():
    ca_config = os.environ.get("PYTORCH_MLU_ALLOC_CONF", "N/A")
    return ca_config


def get_mlu_driver_version(run_lambda):
    return torch_collect_env.run_and_parse_first_match(
        run_lambda, "cnmon", r"Driver (v\d+\.\d+\.\d+)"
    )


def get_mlu_lib_runtime_version(run_lambda, mlu_lib_name):
    mlu_lib_runtime_version = "N/A"
    torch_mlu_path = torch_collect_env.run_and_parse_first_match(
        run_lambda, "pip show torch_mlu", r"Location: (.+)"
    )
    if not torch_mlu_path:
        return mlu_lib_runtime_version

    torch_mlu_so_path = os.path.join(
        torch_mlu_path, "torch_mlu", "_MLUC.cpython-*-*-linux-gnu.so"
    )
    cmd = """
ldd {} | grep '{}.so' | awk '{{print $3}}' | xargs readlink -f | xargs basename | sed 's/lib{}.so.//'
    """.format(
        torch_mlu_so_path, mlu_lib_name, mlu_lib_name
    )
    match = torch_collect_env.run_and_read_all(run_lambda, cmd)
    if match:
        mlu_lib_runtime_version = "v" + match

    return mlu_lib_runtime_version


def get_cndali_runtime_version(run_lambda):
    cndali_runtime_version = "N/A"
    match = torch_collect_env.run_and_parse_first_match(
        run_lambda, "pip list | grep cambricon-dali", r"(\d+\.\d+\.\d+)"
    )
    if match:
        cndali_runtime_version = "v" + match

    return cndali_runtime_version


def get_cncv_runtime_version(run_lambda):
    cncv_runtime_version = "N/A"
    cambricon_dali_path = torch_collect_env.run_and_parse_first_match(
        run_lambda, "pip show cambricon-dali", r"Location: (.+)"
    )
    if not cambricon_dali_path:
        return cncv_runtime_version

    cambricon_dali_so_path = os.path.join(
        cambricon_dali_path, "cambricon", "dali", "libdali.so"
    )
    cmd = """
ldd {} | grep '{}.so' | awk '{{print $3}}' | xargs readlink -f | xargs basename | sed 's/lib{}.so.//'
    """.format(
        cambricon_dali_so_path, "cncv", "cncv"
    )
    match = torch_collect_env.run_and_read_all(run_lambda, cmd)
    if match:
        cncv_runtime_version = "v" + match

    return cncv_runtime_version


def get_env_info():
    run_lambda = torch_collect_env.run
    pip_version, pip_list_output = torch_collect_env.get_pip_packages(run_lambda)

    version_str = torch.__version__
    debug_mode_str = str(torch.version.debug)
    mlu_available_str = str(torch_mlu.mlu.is_available())

    if TORCH_MLU_AVAILABLE:
        torch_mlu_version_str = torch_mlu.__version__
    else:
        torch_mlu_version_str = "N/A"

    sys_version = sys.version.replace("\n", " ")

    return SystemEnv(
        torch_version=version_str,
        torch_mlu_version=torch_mlu_version_str,
        is_debug_build=debug_mode_str,
        os=torch_collect_env.get_os(run_lambda),
        gcc_version=torch_collect_env.get_gcc_version(run_lambda),
        clang_version=torch_collect_env.get_clang_version(run_lambda),
        cmake_version=torch_collect_env.get_cmake_version(run_lambda),
        libc_version=torch_collect_env.get_libc_version(),
        python_version="{} ({}-bit runtime)".format(
            sys_version, sys.maxsize.bit_length() + 1
        ),
        python_platform=torch_collect_env.get_python_platform(),
        caching_allocator_config=get_cachingallocator_config(),
        is_mlu_available=mlu_available_str,
        is_xnnpack_available=torch_collect_env.is_xnnpack_available(),
        mlu_driver_version=get_mlu_driver_version(run_lambda),
        cnrt_runtime_version=get_mlu_lib_runtime_version(run_lambda, "cnrt"),
        cndev_runtime_version=get_mlu_lib_runtime_version(run_lambda, "cndev"),
        cndrv_runtime_version=get_mlu_lib_runtime_version(run_lambda, "cndrv"),
        cnpapi_runtime_version=get_mlu_lib_runtime_version(run_lambda, "cnpapi"),
        cndali_runtime_version=get_cndali_runtime_version(run_lambda),
        cnnl_runtime_verson=get_mlu_lib_runtime_version(run_lambda, "cnnl"),
        mluops_runtime_version=get_mlu_lib_runtime_version(run_lambda, "mluops"),
        cncl_runtime_version=get_mlu_lib_runtime_version(run_lambda, "cncl"),
        cncv_runtime_version=get_cncv_runtime_version(run_lambda),
        cnnlextra_runtime_version=get_mlu_lib_runtime_version(run_lambda, "cnnl_extra"),
        cpu_info=torch_collect_env.get_cpu_info(run_lambda),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=torch_collect_env.get_conda_packages(run_lambda),
    )


env_info_fmt = """
PyTorch version: {torch_version}
Torch-mlu version: {torch_mlu_version}
Is debug build: {is_debug_build}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Caching allocator config: {caching_allocator_config}
Is MLU available: {is_mlu_available}
Is XNNPACK available: {is_xnnpack_available}

MLU driver version: {mlu_driver_version}
cnrt runtime version: {cnrt_runtime_version}
cndev runtime version: {cndev_runtime_version}
cndrv runtime version: {cndrv_runtime_version}
cnpapi runtime version: {cnpapi_runtime_version}
cndali runtime version: {cndali_runtime_version}
cnnl runtime version: {cnnl_runtime_verson}
mlu-ops runtime version: {mluops_runtime_version}
cncl runtime version: {cncl_runtime_version}
cncv runtime version: {cncv_runtime_version}
cnnl-extra runtime version: {cnnlextra_runtime_version}

CPU:
{cpu_info}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag="[prepend]"):
        lines = text.split("\n")
        updated_lines = [tag + line for line in lines]
        return "\n".join(updated_lines)

    def replace_if_empty(text, replacement="No relevant packages"):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])
    mutable_dict["conda_packages"] = replace_if_empty(mutable_dict["conda_packages"])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = prepend(
            mutable_dict["pip_packages"], "[{}] ".format(envinfo.pip_version)
        )
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = prepend(
            mutable_dict["conda_packages"], "[conda] "
        )
    mutable_dict["cpu_info"] = envinfo.cpu_info
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    return pretty_str(get_env_info())


def main():
    print("Collecting environment infomation...")
    output = get_pretty_env_info()
    print(output)

    if hasattr(torch, "utils") and hasattr(torch.utils, "_crash_handler"):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [
                os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)
            ]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg = (
                f"\n*** Detected a minidump at {latest} created on {creation_time}, "
                "if this is related to your bug please include it when you file a report ***"
            )
            print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
