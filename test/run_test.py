#!/usr/bin/env python
# pylint: disable=C0301,W0613,W0611
from __future__ import print_function

import argparse
from datetime import datetime
import os
import shutil
import subprocess
import sys
import copy
import tempfile
import logging

import torch
from torch.utils import cpp_extension
from common_utils import shell, print_to_stderr, gen_err_message

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Note: torch_ops/ & custom_ops/ are dirs to be expanded in get_selected_tests
TESTS = [
    "mlu/test_event",
    "mlu/test_stream",
    "mlu/test_tf32_ctrl",
    "mlu/test_lazy_init",
    "mlu/test_autograd",
    "mlu/test_mlu_cndev_based_avail",
    "mlu/test_pluggable_allocator",
    "torch_ops/",
    "custom_ops/",
    "test_foreach_op/test_foreach_binary",
    "test_foreach_op/test_foreach_unary",
    "test_foreach_op/test_foreach_lerp",
    "torch/test_save_and_load",
    "torch/test_random",
    "distributed/test_distributed",
    "torch/test_pin_memory",
    "torch/test_complex",
    "torch/test_dataloader",
    "torch/test_set_default_type",
    "utils/test_gpu_migration",
    "utils/test_env_migration",
    "utils/test_cnnl_op_exception",
    "utils/test_counter",
    "utils/test_fakecase_print_log",
    "utils/test_utils",
    "utils/test_dumptool",
    "utils/test_collect_env",
    "utils/test_torch_gpu2mlu",
    "utils/test_distributed_timer",
    "utils/test_monkey_patch_ref_leak.py",
    "optimizer/test_torch_fused_adamw",
    "optimizer/test_torch_fused_adam",
    "distributions/test_distributions",
    "fallback/test_fallback",
    "view_chain/test_close_view_chain",
    "view_chain/test_close_view_chain_fuser",
    "view_chain/test_print_view_chain",
    "view_chain/test_multiline_views_graph",
    "view_chain/test_residualway_views_graph",
    "view_chain/test_singleline_views_graph",
    "view_chain/test_view_chain",
    "mlu/test_device",
    "mlu/test_mlu",
    "mlu/test_amp",
    "multiprocessing/test_multiprocessing",
    "profiler/test_profiler",
    "profiler/test_profiler_pmu",
    "profiler/test_profiler_emit_cnpx",
    "profiler/test_profiler_with_config",
    "profiler/test_profiler_record_all",
    "profiler/test_profiler_with_mlugraph",
    "profiler/test_profiler_with_mlugraph_and_record_shapes.py",
    "profiler/test_profiler_with_pmu_and_mlugraph",
    "profiler/test_profiler_cnperf",
    "mlu/test_caching_allocator",
    "cpp_extension/test_mlu_extension",
    "cnnl_gtest",
    "common_gtest",
    "codegen/test_gen_mlu_stubs",
    "storage/test_storage",
    "kineto_gtest",
    "profiler/test_kineto_tb_plugin",
    "cnpx/test_cnpx.py",
    dir_path + "/examples/training/single_card_demo",
    dir_path + "/examples/training/multi_card_demo",
    "api_support_list/test_api_support_list",
]

SINGLE_CARD_SKIP_TEST = [
    dir_path + "/examples/training/multi_card_demo",
    "distributed/test_distributed",
    "utils/test_distributed_timer",
    "mlu/test_mlu",
]

CNNL_BLACKLIST = [
    "torch_ops/",
]

# Case used to generate .xml log, see description inside this file
FAKECASE = "utils/test_fakecase_print_log"


def run_test(test_module, test_directory, options, *extra_unittest_args):
    executable = get_executable_command(options, allow_pytest=True)
    unittest_args = options.additional_unittest_args.copy()
    if options.verbose:
        unittest_args.append(f'-{"v"*options.verbose}')

    log_base = ""
    # If using pytest, replace -f with equivalent -x
    if options.pytest:
        unittest_args = [arg if arg != "-f" else "-x" for arg in unittest_args]
        if options.result_dir != "" and os.path.isdir(options.result_dir):
            log_base = os.path.join(options.result_dir, test_module.replace("/", "_"))
            unittest_args += [f"--junitxml={log_base}.xml"]
    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test_module + ".py"] + unittest_args + list(extra_unittest_args)

    command = executable + argv
    run_env = os.environ.copy()
    run_env["ENABLE_FALLBACK_TO_CPU"] = "0"
    if options.large:
        run_env["TEST_LARGETENSOR"] = "TRUE"
    if log_base:
        ret_code = shell(command, test_directory, run_env, log_base + ".log")
        if not os.path.exists(log_base + ".xml"):
            run_env["FAILED_TEST_MODULE"] = test_module
            run_env["FAILED_LOG_FILE"] = log_base + ".log"
            shell(
                executable + [FAKECASE + ".py", f"--junitxml={log_base}_fakecase.xml"],
                test_directory,
                run_env,
            )
        return ret_code
    else:
        return shell(command, test_directory, run_env)


def get_backend_type(test_module):
    if "cnnl" in test_module:
        return "cnnl"
    else:
        raise RuntimeError("unsupported backend type, currently only support cnnl.")


def test_executable_file(test_module, test_directory, options):
    gtest_dir = os.path.join(test_directory, "../build/bin")
    if "cpp_op_gtest" in test_module:
        gtest_dir = os.path.join(test_directory, "cpp/build/bin")
        gtest_dir = os.path.join(gtest_dir, "op_test")
    elif "cnnl_gtest" in test_module:
        gtest_dir = os.path.join(gtest_dir, "cnnl")
    elif "kineto_gtest" in test_module:
        gtest_dir = os.path.join(test_directory, "../build/kineto_gtest")
    else:
        gtest_dir = os.path.join(gtest_dir, "common")

    total_error_info = []
    total_failed_commands = []
    if os.path.exists(gtest_dir):
        commands = (
            os.path.join(gtest_dir, filename) for filename in os.listdir(gtest_dir)
        )
        for command in commands:
            command = [command]
            log_base = ""
            if options.result_dir != "" and os.path.isdir(options.result_dir):
                log_base = os.path.join(
                    options.result_dir,
                    test_module.replace("/", "_") + "_" + command[0].split("/")[-1],
                )
                command += [f"--gtest_output=xml:{log_base}.xml"]
            if log_base:
                executable = get_executable_command(options, allow_pytest=True)
                run_env = os.environ.copy()
                return_code = shell(command, test_directory, run_env, log_base + ".log")
                if not os.path.exists(log_base + ".xml"):
                    run_env["FAILED_TEST_MODULE"] = (
                        test_module + "/" + command[0].split("/")[-1]
                    )
                    run_env["FAILED_LOG_FILE"] = log_base + ".log"
                    shell(
                        executable
                        + [FAKECASE + ".py", f"--junitxml={log_base}_fakecase.xml"],
                        test_directory,
                        run_env,
                    )
            else:
                return_code = shell(command, test_directory)
            if options.failfast:
                gen_err_message(return_code, command[0], total_error_info)
            elif return_code != 0:
                total_failed_commands.append((command[0], return_code))

    # Print total error message
    print("*********** Gtest : Error Message Summaries **************")
    for err_message in total_error_info:
        logging.error("\033[31;1m {}\033[0m .".format(err_message))
    for cmd, ret in total_failed_commands:
        print(f"command {cmd} failed, return code {ret}")
    print("**********************************************************")

    return 1 if total_failed_commands or total_error_info else 0


def test_with_stdout(test_module, test_directory, options):
    return run_test(test_module, test_directory, options, "-s")


def run_test_with_subprocess_for_mlu(test_module, test_directory, options):
    options_copy = copy.deepcopy(options)
    options_copy.pytest = False
    return run_test(
        test_module,
        test_directory,
        options_copy,
        "--subprocess",
        "--result_dir",
        options.result_dir,
    )


CUSTOM_HANDLERS = {
    "cnnl_gtest": test_executable_file,
    "common_gtest": test_executable_file,
    "cpp_op_gtest": test_executable_file,
    "kineto_gtest": test_executable_file,
    "utils/test_distributed_timer": test_with_stdout,
    "mlu/test_mlu_cndev_based_avail": run_test_with_subprocess_for_mlu,
}


def parse_test_module(test):
    return test.split(".")[0]


class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super(TestChoices, self).__init__(args[0])

    def __contains__(self, item):
        return list.__contains__(self, parse_test_module(item))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        epilog="where TESTS is any of: {}".format(", ".join(TESTS)),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose information and test-by-test results",
    )
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar="TESTS",
        help="select a set of tests to include (defaults to ALL tests)."
        " tests can be specified with module name, module.TestClass"
        " or module.TestClass.test_method",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        choices=TESTS,
        metavar="TESTS",
        default=[],
        help="select a set of tests to exclude",
    )
    parser.add_argument(
        "-f",
        "--first",
        choices=TESTS,
        metavar="TESTS",
        help="select the test to start from (excludes previous tests)",
    )
    parser.add_argument(
        "-l",
        "--last",
        choices=TESTS,
        metavar="TESTS",
        help="select the last test to run (excludes following tests)",
    )
    parser.add_argument(
        "--bring-to-front",
        nargs="+",
        choices=TestChoices(TESTS),
        default=[],
        metavar="TESTS",
        help="select a set of tests to run first. This can be used in situations"
        " where you want to run all tests, but care more about some set, "
        "e.g. after making a change to a specific component",
    )
    parser.add_argument(
        "--only_single_card_test",
        action="store_true",
        help="only run single card tests",
    )
    parser.add_argument(
        "--ignore_cnnl_blacklist",
        action="store_true",
        help="always ignore blacklisted train tests",
    )
    parser.add_argument(
        "additional_unittest_args",
        nargs="*",
        help="additional arguments passed through to unittest, e.g., "
        "python run_test.py -i sparse -- TestSparse.test_factory_size_check",
    )
    parser.add_argument(
        "--large", action="store_true", help="whether to run test cases of large tensor"
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="If true, use `pytest` to execute the tests.",
    )
    parser.add_argument(
        "--result_dir",
        default="",
        help="If result_dir is not empty, generate xml results to the specified directory. "
        "For .py files, xml report is generated only if pytest is enabled.",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="If true, exits immediately upon the first failing testcase."
        "Otherwise, all selected tests are executed regardless of failures.",
    )
    return parser.parse_args()


def get_executable_command(options, allow_pytest):
    executable = [sys.executable]
    if options.pytest:
        if allow_pytest:
            executable += ["-m", "pytest"]
        else:
            print_to_stderr(
                "Pytest cannot be used for this test. Falling back to unittest."
            )
    return executable


def find_test_index(test, selected_tests, find_last_index=False):
    """Find the index of the first or last occurrence of a given test/test module in the list of selected tests.

    This function is used to determine the indices when slicing the list of selected tests when
    ``options.first``(:attr:`find_last_index`=False) and/or ``options.last``(:attr:`find_last_index`=True) are used.

    :attr:`selected_tests` can be a list that contains multiple consequent occurrences of tests
    as part of the same test module, e.g.:

    ```
    selected_tests = ['autograd', 'cuda', **'torch.TestTorch.test_acos',
                     'torch.TestTorch.test_tan', 'torch.TestTorch.test_add'**, 'utils']
    ```

    If :attr:`test`='torch' and :attr:`find_last_index`=False, result should be **2**.
    If :attr:`test`='torch' and :attr:`find_last_index`=True, result should be **4**.

    Arguments:
        test (str): Name of test to lookup
        selected_tests (list): List of tests
        find_last_index (bool, optional): should we lookup the index of first or last
            occurrence (first is default)

    Returns:
        index of the first or last occurance of the given test
    """
    idx = 0
    found_idx = -1
    for t in selected_tests:
        if t.startswith(test):
            found_idx = idx
            if not find_last_index:
                break
        idx += 1
    return found_idx


def exclude_tests(exclude_list, selected_tests, exclude_message=None):
    tests_copy = selected_tests[:]
    for exclude_test in exclude_list:
        for test in tests_copy:
            # Using full match to avoid the problem of
            # similar file names.
            if test == exclude_test:
                if exclude_message is not None:
                    print_to_stderr("Excluding {} {}".format(test, exclude_message))
                selected_tests.remove(test)
    return selected_tests


def get_selected_tests(options):
    selected_tests = options.include

    if options.bring_to_front:
        to_front = set(options.bring_to_front)
        selected_tests = options.bring_to_front + list(
            filter(lambda name: name not in to_front, selected_tests)
        )

    if options.first:
        first_index = find_test_index(options.first, selected_tests)
        selected_tests = selected_tests[first_index:]

    if options.last:
        last_index = find_test_index(options.last, selected_tests, find_last_index=True)
        selected_tests = selected_tests[: last_index + 1]

    selected_tests = exclude_tests(options.exclude, selected_tests)

    print("=========options.ignore_cnnl_blacklist===========")
    if not options.ignore_cnnl_blacklist:
        selected_tests = exclude_tests(CNNL_BLACKLIST, selected_tests)
    print("=========options.ignore_cnnl_blacklist===========")

    selected_copy = selected_tests.copy()
    for selected in selected_copy:
        # TODO(fanshijie): test_distributed.py does not support pytest
        if selected == "distributed/test_distributed" and options.pytest:
            selected_tests = exclude_tests([selected], selected_tests)
            if not options.only_single_card_test:
                selected_tests += ["distributed/test_distributed_wrapper"]
            continue

        if selected in ["torch_ops/", "custom_ops/"]:
            selected_tests += select_current_op_portion(selected)
            selected_tests = exclude_tests([selected], selected_tests)
        elif selected in SINGLE_CARD_SKIP_TEST and options.only_single_card_test:
            selected_tests = exclude_tests([selected], selected_tests)
        else:
            continue

    return selected_tests


"""
    * This function splits testcases in module [torch_ops/, custom_ops/] into
    * CI_PARALLEL_TOTAL number of portions, with currently selected portion
    * index being CI_PARALLEL_INDEX.
    * CI_PARALLEL_TOTAL and CI_PARALLEL_INDEX are env variables set by
    * jenkins pipeline when parallel is used.
    * By default, all testcases are selected.
"""


def select_current_op_portion(module):
    parallel_total = int(os.environ.get("CI_PARALLEL_TOTAL", 1))
    parallel_index = int(os.environ.get("CI_PARALLEL_INDEX", 0))
    op_test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), module)
    all_op_test = []
    for f in os.listdir(op_test_path):
        if f.startswith("test_") and f.endswith(".py"):
            all_op_test += [os.path.join(module, f).split(".")[0]]
    all_op_test = sorted(all_op_test)
    selected_op_test = []
    for index, op_module in enumerate(all_op_test):
        if parallel_index == index % parallel_total:
            selected_op_test.append(op_module)
    return selected_op_test


def main():
    options = parse_args()
    # build cpp gtest
    # TODO(kongweiguang): temporarily sheild this code for testing.
    # if options.ignore_cnnl_blacklist:
    #    subprocess.check_call('cpp/scripts/build_cpp_test.sh')

    test_directory = os.path.dirname(os.path.abspath(__file__))

    selected_tests = get_selected_tests(options)
    total_error_info = []
    return_code_failed = []

    if options.verbose:
        print_to_stderr("Selected tests: {}".format(", ".join(selected_tests)))

    for test in selected_tests:
        test_module = parse_test_module(test)

        # Printing the date here can help diagnose which tests are slow
        print_to_stderr("Running {} ... [{}]".format(test, datetime.now()))
        handler = CUSTOM_HANDLERS.get(test, run_test)
        return_code = handler(test_module, test_directory, options)
        if options.failfast:
            assert isinstance(return_code, int) and not isinstance(
                return_code, bool
            ), "Return code should be an integer"
            gen_err_message(return_code, test, total_error_info)
        else:
            if not isinstance(return_code, int) or return_code != 0:
                return_code_failed.append((test_module, return_code))

    # Print total error message
    print(
        "***************** run_test.py : Error Message Summaries **********************"
    )
    if total_error_info:
        print("***************** Total Error Info **********************")
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        assert False

    if return_code_failed:
        print("***************** Return Code Failed **********************")
        for test_module, return_code in return_code_failed:
            print(f"test_module {test_module}, return code {return_code}")

    print(
        "*******************************************************************************"
    )


if __name__ == "__main__":
    main()
