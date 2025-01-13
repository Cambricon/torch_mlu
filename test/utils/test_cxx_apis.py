from __future__ import print_function
import torch
import torch.nn as nn
import torch_mlu
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import sys
import os
import copy

import time
import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase
import logging

logging.basicConfig(level=logging.DEBUG)


class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        neuware_home_env = os.environ.get("NEUWARE_HOME", None)
        self.assertIsNotNone(
            neuware_home_env,
            "env NEUWARE_HOME needs to be set to run these test cases.",
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_header_file_compilation(self):
        torch_mlu_home = torch_mlu.__file__
        torch_mlu_home = os.path.dirname(os.path.abspath(torch_mlu_home))
        torch_home = torch.__file__
        torch_home = os.path.dirname(os.path.abspath(torch_home))

        mlu_header_file_path = torch_mlu_home + "/csrc"
        torch_header_file_path = torch_home + "/include"
        torch_api_file_path = torch_home + "/include/torch/csrc/api/include"
        neuware_home_path = str(os.environ.get("NEUWARE_HOME")) + "/include"
        mlu_header_file_list = []
        from pathlib import Path

        for path in Path(mlu_header_file_path).rglob("*.h"):
            mlu_header_file_list.append(path.as_posix())
        for path in Path(mlu_header_file_path).rglob("*.hpp"):
            mlu_header_file_list.append(path.as_posix())

        def _compile(
            test_file, mlu_path, torch_path, torch_api_file_path, neuware_path
        ):
            command = (
                "g++ "
                + test_file
                + " "
                + "-I"
                + mlu_path
                + " "
                + "-I"
                + torch_path
                + " "
                + "-I"
                + torch_api_file_path
                + " "
                + "-I"
                + neuware_path
            )
            command += " -o /dev/null"
            import subprocess

            result = subprocess.run(
                command, shell=True, text=True, capture_output=True, check=False
            )

            # Access the exit code
            exit_code = result.returncode

            return exit_code, command

        # test 1: compile each header file using include path from torch wheel, torch_mlu wheel and neuware home to make sure header files
        # available to users only contain public header files.
        for file in mlu_header_file_list:
            result_code, command = _compile(
                file,
                mlu_header_file_path,
                torch_header_file_path,
                torch_api_file_path,
                neuware_home_path,
            )
            try:
                self.assertEqual(result_code, 0)
            except Exception as e:
                raise AssertionError(
                    f"compile command {command}, may be private header files are included in above header file."
                ) from e

        def _extract_gcc_include_paths(gcc_executable="gcc"):
            import subprocess

            # Run the GCC command to list include paths
            result = subprocess.run(
                [gcc_executable, "-E", "-v", "-"],
                stdin=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                text=True,
            )

            # Extract the relevant section of the output
            output = result.stderr
            include_paths = []
            include_section = False

            for line in output.splitlines():
                if line.startswith("#include <...> search starts here:"):
                    include_section = True
                    continue
                if include_section:
                    if line.startswith("End of search list."):
                        break
                    # Add the trimmed line as an include path
                    include_paths.append(line.strip())

            return include_paths

        def _extract_cpp_function_symbols(
            header_file, torch_mlu_include_path, gcc_include_path
        ):
            import clang.cindex
            from clang.cindex import CursorKind

            """
            Extract function symbols from a C++ header file, excluding inline functions
            and ensuring extracted symbols include parameters.
            """
            # Initialize Clang Index
            index = clang.cindex.Index.create()

            gcc_path_list = []
            for path in gcc_include_path:
                gcc_path_list.append(f"-I{path}")

            torch_mlu_path_list = []
            for path in torch_mlu_include_path:
                torch_mlu_path_list.append(f"-I{path}")

            abi_version = (int)(torch.compiled_with_cxx11_abi())

            # Parse the C++ header file
            translation_unit = index.parse(
                header_file,
                args=[
                    "-x",
                    "c++",
                    "-std=c++17",
                    f"-D_GLIBCXX_USE_CXX11_ABI={abi_version}",
                ]
                + torch_mlu_path_list
                + gcc_path_list,
            )

            # List to store extracted function symbols
            functions = []

            # Function to recursively visit AST nodes
            def visit_node(node):
                if node.kind == CursorKind.FUNCTION_DECL:
                    dummy_ = str(node.location)
                    if node.linkage == clang.cindex.LinkageKind.INTERNAL:
                        return
                    if node.is_definition():
                        return
                    if str(node.location._data[0]) == header_file:
                        # Check if the function has parameters
                        if len(list(node.get_arguments())) > 0:
                            mangled_func_name = node.mangled_name
                            functions.append(f"{mangled_func_name}")

                # Recursively visit children
                for child in node.get_children():
                    visit_node(child)

            # Start visiting from the root node
            visit_node(translation_unit.cursor)

            return functions

        def _grep_symbol_from_dynamic_library(symbol, library_path):
            command = "nm -g " + library_path + " | grep " + symbol + " | wc -l"
            import subprocess

            result = subprocess.run(
                command, shell=True, text=True, capture_output=True, check=False
            )

            return int(result.stdout), symbol

        # test2: Search each function that appears in the header file in the dynamic link library
        # to ensure that the functions exposed to the user are defined in the dynamic link library.
        gcc_include_path = _extract_gcc_include_paths()
        torch_mlu_include_path = [
            mlu_header_file_path,
            torch_header_file_path,
            torch_api_file_path,
            neuware_home_path,
        ]
        for file in mlu_header_file_list:
            functions = _extract_cpp_function_symbols(
                file, torch_mlu_include_path, gcc_include_path
            )
            mlu_dynamic_library_path = torch_mlu_home + "/csrc/lib/libtorch_mlu.so"
            for function in functions:
                result, symbol = _grep_symbol_from_dynamic_library(
                    function, mlu_dynamic_library_path
                )
                try:
                    self.assertGreater(result, 0)
                except Exception as e:
                    raise AssertionError(
                        f"failed to find {symbol} in library libtorch_mlu.so, the symbol is found in file {file}, perhaps you need to add the TORCH_MLU_API macro to the definition of this symbol."
                    ) from e


if __name__ == "__main__":
    unittest.main()
