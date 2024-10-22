#!/usr/bin/env python
from __future__ import print_function
import os
import re
import sys
import sysconfig
import traceback
import textwrap
import stat
import glob
import shutil
import json
import subprocess
import torch

import distutils.ccompiler  # pylint: disable=C0411
import distutils.command.clean  # pylint: disable=C0411
import setuptools.command.install  # pylint: disable=C0411

from setuptools import setup, find_packages, distutils  # pylint: disable=C0411
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
)  # pylint: disable=C0411


################################################################################
# Parameters parsed from environment
################################################################################
RUN_BUILD_CORE_LIBS = True
RUN_AUTO_GEN_TORCH_MLU_CODE = True
RUN_BUILD_ASAN_CHECK = False
RUN_BUILD_WARNING_CHECK = True

RUN_BUILD_USE_PYTHON = bool(
    (os.getenv("USE_PYTHON") is None)
    or (os.getenv("USE_PYTHON").upper() not in ["OFF", "0", "NO", "FALSE", "N"])
)

RUN_BUILD_USE_BANG = bool(
    (os.getenv("USE_BANG") is None)
    or (os.getenv("USE_BANG").upper() not in ["OFF", "0", "NO", "FALSE", "N"])
)

RUN_BUILD_USE_MLUOP = bool(
    (os.getenv("USE_MLUOP") is None)
    or (os.getenv("USE_MLUOP").upper() not in ["OFF", "0", "NO", "FALSE", "N"])
)

for i, arg in enumerate(sys.argv):
    if arg == "clean":
        RUN_BUILD_CORE_LIBS = False
        RUN_AUTO_GEN_TORCH_MLU_CODE = False

# Get the current path, core library paths and neuware path
cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch_mlu", "csrc", "lib")

# NEUWARE_HOME env must be set before compiling
if not os.getenv("NEUWARE_HOME"):
    print(
        "[Error] NEUWARE_HOME Environment Variable has not been set,",
        "Please firstly get and install the Cambricon Neuware Package,",
        "then use NEUWARE_HOME to point it!",
    )
    sys.exit()

# Get Pytorch Dir
base_dir = os.path.dirname(os.path.abspath(__file__))


def get_pytorch_dir():
    try:
        import torch

        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


pytorch_dir = get_pytorch_dir()

# lib/pythonx.x/site-packages
rel_site_packages = distutils.sysconfig.get_python_lib(prefix="")
# full absolute path to the dir above
full_site_packages = distutils.sysconfig.get_python_lib()
# full absolute path to installed torch cmake path
torch_cmake_path = torch.utils.cmake_prefix_path

# Define the compile and link options
extra_link_args = []
extra_compile_args = []


# Check env flag
def _check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def _check_env_off_flag(name, default=""):
    return os.getenv(name, default).upper() in ["OFF", "0", "NO", "FALSE", "N"]


# make relative rpath
def make_relative_rpath(path):
    return "-Wl,-rpath,$ORIGIN/" + path


# Generate parts of header/source files in torch_mlu automatically
def gen_torch_mlu_code():
    order = "python -m codegen.gen_mlu_stubs --source_yaml ./codegen/mlu_functions.yaml"
    if RUN_BUILD_USE_BANG:
        order += " --use_bang"
    if RUN_BUILD_USE_MLUOP:
        order += " --use_mluop"
    os.system(order)


# Calls build_torch_mlu_lib.sh with the corrent env variables
def build_libs():
    build_libs_cmd = ["bash", os.path.join("..", "scripts", "build_torch_mlu_lib.sh")]
    my_env = os.environ.copy()

    my_env["CMAKE_INSTALL"] = "make install"

    cmake_prefix_path = full_site_packages + ";" + torch_cmake_path
    if "CMAKE_PREFIX_PATH" in my_env:
        cmake_prefix_path = my_env["CMAKE_PREFIX_PATH"] + ";" + cmake_prefix_path
    my_env["CMAKE_PREFIX_PATH"] = cmake_prefix_path

    my_env["PYTORCH_WHEEL_DIR"] = pytorch_dir
    my_env["TORCH_MLU_SOURCE_PATH"] = base_dir

    # Keep the same compile and link args between setup.py and build_torch_mlu_lib.sh
    my_env["EXTRA_COMPILE_ARGS"] = " ".join(extra_compile_args)
    my_env["EXTRA_LINK_ARGS"] = " ".join(extra_link_args)

    # set up the gtest compile runtime environment.
    my_env["BUILD_TEST"] = "ON" if _check_env_flag("BUILD_TEST") else "OFF"
    my_env["USE_PYTHON"] = "OFF" if _check_env_off_flag("USE_PYTHON") else "ON"
    my_env["USE_BANG"] = "OFF" if _check_env_off_flag("USE_BANG") else "ON"
    my_env["USE_MLUOP"] = "OFF" if _check_env_off_flag("USE_MLUOP") else "ON"
    my_env["USE_CNCL"] = "OFF" if _check_env_off_flag("USE_CNCL") else "ON"
    my_env["USE_PROFILE"] = "OFF" if _check_env_off_flag("USE_PROFILE") else "ON"
    my_env["USE_MAGICMIND"] = "ON" if _check_env_flag("USE_MAGICMIND") else "OFF"

    # python environment
    cmake_python_library = "{}/{}".format(
        sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("INSTSONAME")
    )
    cmake_python_include_dir = sysconfig.get_path("include")
    my_env["PYTHON_EXECUTABLE"] = sys.executable
    my_env["PYTHON_LIBRARY"] = cmake_python_library
    my_env["PYTHON_INCLUDE_DIR"] = cmake_python_include_dir

    # ABI version
    abi_version = (int)(torch.compiled_with_cxx11_abi())
    my_env["GLIBCXX_USE_CXX11_ABI"] = str(abi_version)

    try:
        os.mkdir("build")
    except OSError:
        pass

    kwargs = {"cwd": "build"}

    if subprocess.call(build_libs_cmd, env=my_env, **kwargs) != 0:
        print(
            "Failed to run '{}'".format(" ".join(build_libs_cmd))
        )  # pylint: disable=C0209
        sys.exit(1)


class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)


class Build(BuildExtension):
    def run(self):
        # Run the original BuildExtension first. We need this before building
        # the tests.
        BuildExtension.run(self)


class Clean(distutils.command.clean.clean):
    def run(self):
        try:
            with open(".gitignore", "r") as f:  # pylint: disable=W1514
                ignores = f.read()
                pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
                for wildcard in filter(None, ignores.split("\n")):
                    match = pat.match(wildcard)
                    if match:
                        if match.group(1):
                            # Marker is found and stop reading .gitignore.
                            break
                        # Ignore lines which begin with '#'.
                    else:
                        for filename in glob.glob(wildcard):
                            try:
                                os.remove(filename)
                            except OSError:
                                shutil.rmtree(filename, ignore_errors=True)
        except OSError:
            shutil.rmtree("build", ignore_errors=True)
        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


# Configuration for Build the Project.
main_libraries = ["torch_mlu_python"]
include_dirs = []
library_dirs = []

# Fetch the sources to be built.
torch_mlu_sources = glob.glob("torch_mlu/csrc/stub.cpp")

# include head files
include_dirs += [
    base_dir,
    os.path.join(pytorch_dir, "include"),
]

# include lib files
library_dirs.append(lib_path)

extra_compile_args += [
    "-std=c++17",
    "-pthread",
    "-Wno-sign-compare",
    "-Wno-deprecated-declarations",
    "-Wno-return-type",
    "-Wno-write-strings",
    "-Wno-stringop-overflow",
    "-Werror",
]

DEBUG = _check_env_flag("DEBUG")
if RUN_BUILD_ASAN_CHECK:
    # To get a reasonable performace add -O1 or higher.
    # run executable with LD_PRELOAD=path/to/asan/runtime/lib
    extra_compile_args += [
        "-O1",
        "-g",
        "-DDEBUG",
        "-fsanitize=address",
        "-fno-omit-frame-pointer",
    ]
elif DEBUG:
    extra_compile_args += ["-O0", "-g", "-DDEBUG"]
else:
    extra_compile_args += ["-O3"]

TEST_COVERAGE = _check_env_flag("TEST_COVERAGE")
if TEST_COVERAGE:
    extra_compile_args += ["-fprofile-arcs", "-ftest-coverage"]
    extra_link_args += ["-fprofile-arcs", "-ftest-coverage"]
    # to test coverage, these args are necessary

config_dir = os.path.join(cwd, "cmake")
modules_dir = os.path.join(cwd, "torch_mlu/share/cmake/TorchMLU/modules")
if not os.path.exists(modules_dir):
    os.makedirs(modules_dir)
shutil.copy(base_dir + "/cmake/modules/FindCNNL.cmake", modules_dir)
shutil.copy(base_dir + "/cmake/modules/FindCNRT.cmake", modules_dir)
shutil.copy(base_dir + "/cmake/modules/FindCNDRV.cmake", modules_dir)
shutil.copy(base_dir + "/cmake/modules/FindCNPAPI.cmake", modules_dir)

if _check_env_flag("USE_MLUOP"):
    shutil.copy(base_dir + "/cmake/modules/FindMLUOP.cmake", modules_dir)

# Replace pre-commit of .git to use cpplint
if os.path.exists(os.path.join(cwd, ".git")):
    shutil.copyfile(
        base_dir + "/scripts/hooks/pre-commit", base_dir + "/.git/hooks/pre-commit"
    )
    # Set the file permission to 755
    os.chmod(
        base_dir + "/.git/hooks/pre-commit",
        stat.S_IRWXU + stat.S_IROTH + stat.S_IXOTH + stat.S_IRGRP + stat.S_IXGRP,
    )

# Generate parts of torch_mlu code
if RUN_AUTO_GEN_TORCH_MLU_CODE:
    gen_torch_mlu_code()

# Build torch_mlu Core Libs
if RUN_BUILD_CORE_LIBS:
    build_libs()

if not RUN_BUILD_USE_PYTHON:
    sys.exit(0)

json_file = os.path.join(cwd, "scripts/release", "build.property")
torch_mlu_version = "unknown"
if os.path.isfile(json_file):
    with open(json_file, "r") as f:  # pylint: disable=W1514
        json_dict = json.load(f)
        # When setuptools >= 66.0.0 version, the previous torch_mlu_version name
        # can't pass the format check(https://github.com/pypa/setuptools/issues/3772),
        # so we replace the '-' with '+' in the name to resolve this problem.
        torch_mlu_version = json_dict["version"].replace("-", "+").strip()

try:
    _git_version = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_dir)
        .decode("ascii")
        .strip()
    )
except (OSError, subprocess.CalledProcessError):
    _git_version = "unknown"

version_file_path = os.path.join(base_dir, "torch_mlu", "version.py")
os.makedirs(os.path.dirname(version_file_path), exist_ok=True)
with open(version_file_path, "w", encoding="utf-8") as f:
    f.write(
        textwrap.dedent(
            f"""\
            # This file is generated by setup.py. DO NOT EDIT!

            git_version = "{_git_version}"
            mlu_version = "{torch_mlu_version}"
            """
        )
    )

# Setup
setup(
    name="torch_mlu",
    version=torch_mlu_version,
    description="MLU bridge for PyTorch",
    packages=find_packages(exclude=["tools", "tools.*", "codegen", "codegen.*"]),
    ext_modules=[
        CppExtension(
            "torch_mlu._MLUC",
            libraries=main_libraries,
            sources=torch_mlu_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + [make_relative_rpath("csrc/lib")],
        ),
    ],
    package_data={
        "torch_mlu": [
            "csrc/lib/*.so*",
            "share/cmake/TorchMLU/TorchMLUConfig.cmake",
            "share/cmake/TorchMLU/modules/*.cmake",
            "csrc/aten/cnnl/cnnlHandle.h",
            "csrc/aten/cnnl/cnnlTensorDescriptors.h",
            "csrc/aten/utils/cnnl_util.h",
            "csrc/aten/utils/tensor_util.h",
            "csrc/aten/utils/types.h",
            "csrc/aten/utils/exceptions.h",
            "csrc/framework/core/device.h",
            "csrc/framework/core/guard_impl.h",
            "csrc/framework/core/mlu_guard.h",
            "csrc/framework/core/stream_guard.h",
            "csrc/framework/core/MLUEvent.h",
            "csrc/framework/core/MLUStream.h",
            "csrc/framework/core/caching_allocator.h",
            "csrc/framework/generator/generator_impl.h",
            "csrc/framework/distributed/process_group_cncl.hpp",
            "csrc/utils/Export.h",
            "csrc/aten/mluop/mluopHandle.h",
            "csrc/aten/mluop/mluopUtils.h",
            "csrc/aten/mluop/mluopCommonDescriptors.h",
            "csrc/aten/mluop/mluopTensorDescriptors.h",
            "csrc/aten/mluop/mluopDescriptors.h",
        ],
    },
    cmdclass={
        "build_ext": Build,
        "clean": Clean,
        "install": install,
    },
)
