import os
import glob
import sys
import shutil

from setuptools.dist import Distribution

from setuptools import setup, find_packages
from torch_mlu.utils.cpp_extension import MLUExtension, BuildExtension


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "mlu_custom_ext", "src")
    sources = glob.glob(os.path.join(extensions_dir, "*.cpp")) + glob.glob(
        os.path.join(extensions_dir, "mlu", "*.mlu")
    )

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    mlu_extension = MLUExtension(
        name="mlu_custom_ext._C",
        sources=sorted(sources),
        include_dirs=include_dirs,
        verbose=True,
        extra_cflags=["-w"],
        extra_link_args=["-w"],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
            ],
            "cncc": ["-O3", "-I{}".format(extensions_dir)],
        },
    )

    dist = Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    if dist.script_args == ["clean"]:
        if os.path.exists(os.path.abspath("build")):
            shutil.rmtree("build")

    setup(
        name="mlu_custom_ext",
        version="0.1",
        packages=find_packages(),
        ext_modules=[mlu_extension],
        cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    )
