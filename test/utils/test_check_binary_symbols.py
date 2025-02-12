from __future__ import print_function
import torch
import sys
import os
import concurrent.futures
import distutils.sysconfig
import itertools
import functools
import re
from pathlib import Path

import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
import logging

logging.basicConfig(level=logging.DEBUG)

# We also check that there are [not] cxx11 symbols in libtorch_mlu
#
# To check whether it is using cxx11 ABI, check non-existence of symbol:
PRE_CXX11_SYMBOLS = (
    "std::basic_string<",
    "std::list",
)
# To check whether it is using pre-cxx11 ABI, check non-existence of symbol:
CXX11_SYMBOLS = (
    "std::__cxx11::basic_string",
    "std::__cxx11::list",
)
# NOTE: Checking the above symbols in all namespaces doesn't work, because
# devtoolset7 always produces some cxx11 symbols even if we build with old ABI,
# and CuDNN always has pre-cxx11 symbols even if we build with new ABI using gcc 5.4.
# Instead, we *only* check the above symbols in the following namespaces:
LIBTORCH_NAMESPACE_LIST = (
    "at::",
    "torch_mlu::",
)

LIBTORCH_CXX11_PATTERNS = [
    re.compile(f"{x}.*{y}")
    for (x, y) in itertools.product(LIBTORCH_NAMESPACE_LIST, CXX11_SYMBOLS)
]

LIBTORCH_PRE_CXX11_PATTERNS = [
    re.compile(f"{x}.*{y}")
    for (x, y) in itertools.product(LIBTORCH_NAMESPACE_LIST, PRE_CXX11_SYMBOLS)
]


class TestBinarySymbols(TestCase):
    @functools.lru_cache(100)
    def get_symbols(self, lib):
        from subprocess import check_output

        lines = check_output(f'nm "{lib}"|c++filt', shell=True)
        return [x.split(" ", 2) for x in lines.decode("latin1").split("\n")[:-1]]

    def grep_symbols(self, lib, patterns):
        def _grep_symbols(symbols, patterns):
            rc = []
            for s_addr, s_type, s_name in symbols:
                for pattern in patterns:
                    if pattern.match(s_name):
                        rc.append(s_name)
                        continue
            return rc

        all_symbols = self.get_symbols(lib)
        num_workers = 8
        chunk_size = (len(all_symbols) + num_workers - 1) // num_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [
                executor.submit(
                    _grep_symbols,
                    all_symbols[i * chunk_size : (i + 1) * chunk_size],
                    patterns,
                )
                for i in range(num_workers)
            ]
            return sum((x.result() for x in tasks), [])

    def check_lib_symbols_for_abi_correctness(self, lib, pre_cxx11_abi=True):
        print(f"lib: {lib}")
        cxx11_symbols = self.grep_symbols(lib, LIBTORCH_CXX11_PATTERNS)
        pre_cxx11_symbols = self.grep_symbols(lib, LIBTORCH_PRE_CXX11_PATTERNS)
        num_cxx11_symbols = len(cxx11_symbols)
        num_pre_cxx11_symbols = len(pre_cxx11_symbols)
        print(f"num_cxx11_symbols: {num_cxx11_symbols}")
        print(f"num_pre_cxx11_symbols: {num_pre_cxx11_symbols}")
        if pre_cxx11_abi:
            if num_cxx11_symbols > 0:
                raise RuntimeError(
                    f"Found cxx11 symbols, but there shouldn't be any, see: {cxx11_symbols[:100]}"
                )
            if num_pre_cxx11_symbols < 10:
                raise RuntimeError("Didn't find enough pre-cxx11 symbols.")
        else:
            if num_pre_cxx11_symbols > 0:
                raise RuntimeError(
                    f"Found pre-cxx11 symbols, but there shouldn't be any, see: {pre_cxx11_symbols[:100]}"
                )
            if num_cxx11_symbols < 10:
                raise RuntimeError("Didn't find enought cxx11 symbols")

    # @unittest.skip("not test")
    @testinfo()
    def test_check_binary_symbols(self):
        install_root = Path(distutils.sysconfig.get_python_lib()) / "torch_mlu"
        libtorch_mlu_path = install_root / "csrc" / "lib" / "libtorch_mlu.so"
        pre_cxx11_abi = not torch.compiled_with_cxx11_abi()
        self.check_lib_symbols_for_abi_correctness(libtorch_mlu_path, pre_cxx11_abi)


if __name__ == "__main__":
    unittest.main()
