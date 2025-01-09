# pylint: disable=C0301,C0305,W0511,W0212,W0621,W1201,W0613,C0123,R0201,R1722,R1711,W0102,W0612,C0200,R0124,R0912
from __future__ import print_function
import unittest
import time
import argparse
import sys
import os
import functools
import gc
import subprocess
import re
import signal
import logging
import inspect
import json
import traceback
import copy
import random
import warnings
import threading
from functools import wraps
from numbers import Number
from typing import cast, Optional
from collections import OrderedDict, Counter
import pandas as pd
import numpy as np
import torch
import contextlib
from torch.testing._internal.common_utils import TestCase as BaseTestCase
import torch_mlu
from torch.utils._mode_utils import no_dispatch
from contextlib import closing, contextmanager
from statistics import mean
import __main__

try:
    import psutil  # type: ignore[import]

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--subprocess", action="store_true", help="whether to run each test in a subprocess"
)
parser.add_argument(
    "--large", action="store_true", help="whether to run test cases of large tensor"
)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--accept", action="store_true")
parser.add_argument(
    "--save-xml",
    action="store_true",
    help="If true, save xml result in result_dir",
)
parser.add_argument(
    "--result_dir",
    default="",
    help="If result_dir is not empty, generate xml results to the specified directory.",
)
args, remaining = parser.parse_known_args()
TEST_IN_SUBPROCESS = args.subprocess
TEST_LARGETENSOR = args.large or os.environ.get(
    "TEST_LARGETENSOR", "FALSE"
).upper() in ["TRUE", "YES", "ON"]
SEED = args.seed
TEST_WITH_PERFTOOL = os.environ.get("TEST_WITH_PERFTOOL", "FALSE").upper() in [
    "TRUE",
    "YES",
    "ON",
]
UNITTEST_ARGS = [sys.argv[0]] + remaining

SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}


@functools.lru_cache
def get_cycles_per_ms() -> float:
    """Measure and return approximate number of cycles per millisecond for torch.mlu._sleep"""

    def measure() -> float:
        start = torch.mlu.Event(enable_timing=True)
        end = torch.mlu.Event(enable_timing=True)
        start.record()
        torch.mlu._sleep(1000000)
        end.record()
        end.synchronize()
        cycles_per_ms = 1000000 / start.elapsed_time(end)
        return cycles_per_ms

    # use average time to reduce the effect of hardware scheduling on results
    num = 10
    vals = []
    for _ in range(num):
        vals.append(measure())
    vals = sorted(vals)
    return mean(vals[2 : num - 2])


def read_openblas_info():
    try:
        path1 = os.popen(
            "find $(dirname $(dirname $(which python))) -name libtorch_cpu.so"
        ).readlines()
        path2 = os.popen(
            "find $(dirname $(dirname $(which python))) -name torch.egg-link"
        ).readlines()
        if (
            path1
            and os.popen("ldd " + path1[0].strip() + " | grep libopenblas").readlines()
        ):
            # setup pytorch using 'install' command
            return True
        elif path2:  # setup pytorch using 'develop' command
            path_pytorch = os.popen("cat " + path2[0].strip()).readlines()
            path_so = os.popen(
                "find " + path_pytorch[0].strip() + "/build/lib | grep libtorch_cpu.so"
            ).readlines()
            if (
                path_so
                and os.popen(
                    "ldd " + path_so[0].strip() + " | grep libopenblas"
                ).readlines()
            ):
                return True
        return False
    except:
        return False


def read_card_info():
    try:
        result = subprocess.run(
            ["cnmon", "info", "-c", "0", "-t"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        compile = re.compile("MLU5[0-9]*-?[0-9a-zA-Z]*")
        card_name = compile.findall(result.stdout)[0]
    except:
        return False
    else:
        return True


def skipBFloat16IfNotSupport():
    def loader(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception:
                logging.info(
                    "\033[1;35m {} is not support bfloat16. \033[0m".format(
                        type(args[0]).__name__
                    )
                )
                pass

        return wrapper

    return loader


def getGCCVersion():
    import subprocess

    output = subprocess.check_output(["gcc", "--version"], text=True)
    first_line = output.splitlines()[0]
    version = first_line.split()[-1]
    try:
        parts = list(map(int, version.split(".")))
        return {
            "major": parts[0],
            "minor": parts[1] if len(parts) > 1 else 0,
            "patch": parts[2] if len(parts) > 2 else 0,
        }
    except Exception as e:
        print("An error occured when get gcc version: {e}")
        return []


def skipDtypeNotSupport(*dtypes):
    def loader(func):
        def wrapper(*args, **kwargs):
            for dtype in dtypes:
                try:
                    func(*args, **kwargs, type=dtype)
                except Exception as e:
                    if "CNNL_STATUS_BAD_PARAM" in str(e):
                        logging.info(
                            "\033[1;35m {} does not support {}. \033[0m".format(
                                type(args[0]).__name__, str(dtype)
                            )
                        )
                        pass

        return wrapper

    return loader


def _check_module_exists(name):
    import importlib  # pylint: disable= C0415
    import importlib.util  # pylint: disable= C0415

    spec = importlib.util.find_spec(name)
    return spec is not None


TEST_NUMPY = _check_module_exists("numpy")
if TEST_NUMPY:
    # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
    numpy_to_torch_dtype_dict = {
        bool: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
    }

    # Dict of torch dtype -> NumPy dtype
    torch_to_numpy_dtype_dict = {
        value: key for (key, value) in numpy_to_torch_dtype_dict.items()
    }


def gen_err_message(return_code, test_module, total_error_info):
    # Generate error messages based on the return code of the child process.
    if return_code != 0:
        message = "{} failed!".format(test_module)
        if return_code < 0:
            # subprocess.Popen returns the child process' exit signal as
            # return code -N, where N is the signal number.
            signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
            message += " Received signal: {}".format(signal_name)
            total_error_info.append(message)
            logging.error("\033[31;1m {}\033[0m .".format(message))
        elif return_code == 2:
            raise KeyboardInterrupt
        else:
            total_error_info.append(message)
        assert False, message


def print_to_stderr(message):
    print(message, file=sys.stderr)


def shell(command, cwd=None, env=None, fail_log=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    err_msg = "Command to shell should be a list or tuple of tokens"
    assert not isinstance(command, str), err_msg
    p = subprocess.Popen(
        command,
        universal_newlines=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if fail_log else None,
        stderr=subprocess.STDOUT if fail_log else None,
    )  # pylint: disable= R1732
    stdout, _ = p.communicate()
    try:
        ret_code = p.wait()
        if fail_log:
            print(stdout)
            if ret_code != 0:
                with open(fail_log, "w", encoding="utf-8") as f:
                    f.write(stdout)
        return ret_code
    except KeyboardInterrupt:
        # Give `p` a chance to handle KeyboardInterrupt. Without this,
        # `pytest` can't print errors it collected so far upon KeyboardInterrupt.
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            # return exit_status
            return int(2)
        else:
            p.kill()
            return int(2)
    except:  # noqa E722, copied from python core library
        p.kill()
        raise
    finally:
        # Always call p.wait() to ensure exit
        p.wait()


def _get_test_report_path():
    test_source = "python-unittest" if args.result_dir == "" else args.result_dir
    return test_source


def run_tests(argv=UNITTEST_ARGS):
    if TEST_IN_SUBPROCESS:
        suite = unittest.TestLoader().loadTestsFromModule(__main__)
        test_cases = []

        def add_to_test_cases(suite_or_case):
            if isinstance(suite_or_case, unittest.TestCase):
                test_cases.append(suite_or_case)
            else:
                for element in suite_or_case:
                    add_to_test_cases(element)

        add_to_test_cases(suite)
        failed_tests = []
        for case in test_cases:
            test_case_full_name = case.id().split(".", 1)[1]
            exitcode = shell(
                [sys.executable]
                + argv
                + ["--save-xml"]
                + ["--result_dir", args.result_dir]
                + [test_case_full_name]
            )
            if exitcode != 0:
                failed_tests.append(test_case_full_name)

        assert len(failed_tests) == 0, "{} unit test(s) failed:\n\t{}".format(
            len(failed_tests), "\n\t".join(failed_tests)
        )
    else:
        if args.save_xml and args.result_dir != "":
            import xmlrunner  # type: ignore[import]
            from xmlrunner.result import _XMLTestResult  # type: ignore[import]

            class XMLTestResultVerbose(_XMLTestResult):
                """
                Adding verbosity to test outputs:
                by default test summary prints 'skip',
                but we want to also print the skip reason.
                GH issue: https://github.com/pytorch/pytorch/issues/69014

                This works with unittest_xml_reporting<=3.2.0,>=2.0.0
                (3.2.0 is latest at the moment)
                """

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def addSkip(self, test, reason):
                    super().addSkip(test, reason)
                    for c in self.callback.__closure__:
                        if (
                            isinstance(c.cell_contents, str)
                            and c.cell_contents == "skip"
                        ):
                            # this message is printed in test summary;
                            # it stands for `verbose_str` captured in the closure
                            c.cell_contents = f"skip: {reason}"

                def printErrors(self) -> None:
                    super().printErrors()
                    self.printErrorList("XPASS", self.unexpectedSuccesses)

            test_report_path = _get_test_report_path()
            print(f"test_report_path {test_report_path}")
            unittest.main(
                argv=argv,
                testRunner=xmlrunner.XMLTestRunner(
                    output=test_report_path,
                    verbosity=2,
                    resultclass=XMLTestResultVerbose,
                ),
            )
        else:
            unittest.main(argv=argv)


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.mlu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def ops_perftool(test_file_name, test_func_name, save_orig_trace=False):
    """Add TORCH_MLU perftool to generate op's kernel calls report if environ-variable TEST_WITH_PERFTOOL is True.

    Arguments:
    test_file_name -- current op's testing file name without extension (e.g., 'test_abs')
    test_func_name -- current op's testing case name (e.g., 'test_abs_contiguous')
    save_orig_trace -- whether saving original trace file of profiler (default False)

    Outputs:
    op's kernel calls report will be saved under '$PWD/torch_ops/ops_kernel_report' dirctory.
    op's original trace file of profiler will be saved under '$PWD/tmp' dirctory if args@save_orig_trace is True.
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _gen_cur_case(orig_trace):
                """Generate a filtered dictionary from original trace file of current case."""
                with open(orig_trace, "r") as tfile:
                    # json decode error with ".0"
                    dic_orig = json.loads(
                        tfile.read().replace(
                            '"memory bandwidth (GB/s)": .0',
                            '"memory bandwidth (GB/s)": 0.0',
                        )
                    )
                evts_lst = [
                    evt
                    for evt in dic_orig["traceEvents"]
                    if evt["name"].startswith("cnnl")
                    or evt["name"] in ["Memcpy HtoD", "Memcpy DtoH"]
                ]
                name_lst = [evt["name"] for evt in evts_lst]
                cnter = Counter(name_lst)
                dic = OrderedDict()
                dic["CaseName"] = test_func_name
                dic["Events"] = [
                    {"Name": k, "# of Calls": val} for k, val in cnter.items()
                ]
                return dic

            def _update_report(dic_case, report_dir, test_file_name):
                """Update current case into the corresponding op's target json report.
                We need to guarantee the consistency of card number for all cases in each test file.
                """
                device_count = torch.mlu.device_count()
                target_json = report_dir + "/" + test_file_name + ".json"
                if os.path.exists(target_json):
                    with open(target_json, "r") as infile:
                        dic_file = json.load(infile)
                    # remove all other cases' report if "device_count" has changed!
                    if dic_file["device_count"] != device_count:
                        dic_file = {
                            "device_count": device_count,
                            test_file_name: [dic_case],
                        }
                    else:
                        for idx, case in enumerate(dic_file[test_file_name]):
                            if dic_case["CaseName"] == case["CaseName"]:
                                dic_file[test_file_name][idx].update(
                                    {"Events": dic_case["Events"]}
                                )
                                break
                            if idx == len(dic_file[test_file_name]) - 1:
                                dic_file[test_file_name].append(dic_case)
                else:
                    dic_file = {
                        "device_count": device_count,
                        test_file_name: [dic_case],
                    }
                with open(target_json, "w") as outfile:
                    json.dump(dic_file, outfile, indent=4)
                logging.info("Save perftool report " + target_json + " successfully.")

            if TEST_WITH_PERFTOOL:
                # fix random seeds to avoid kernel calls changing
                setup_seed(seed=SEED)
                # open profiler
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.MLU,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    with_flops=False,
                ) as prof:
                    func(*args, **kwargs)
                # make all dirs
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                report_dir = cur_dir + "/torch_ops/ops_kernel_report"
                log_dir = cur_dir + "/tmp"
                if not os.path.exists(report_dir):
                    os.makedirs(report_dir)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                # export original trace file
                orig_trace_file = (
                    log_dir
                    + "/"
                    + test_file_name
                    + "_py_"
                    + test_func_name
                    + ".pt.trace.json"
                )
                prof.export_chrome_trace(orig_trace_file)
                # get filtered dic of current case
                dic = _gen_cur_case(orig_trace_file)
                # update current case into report file
                _update_report(dic, report_dir, test_file_name)
                # remove the temporary original trace file
                if not save_orig_trace:
                    os.system("rm " + orig_trace_file)
            else:
                # without profiler
                func(*args, **kwargs)

        return wrapper

    return inner


def testinfo(pre_message="\nTest case:", post_message="Test Time:"):
    def loader(func):
        test_file_name = ""
        test_func_name = ""
        if TEST_WITH_PERFTOOL:
            module_name = inspect.getmodule(func).__name__
            if module_name == "__main__":
                # unittest
                test_file_name = os.path.basename(sys.argv[0]).split(".")[0]
            else:
                # pytest
                test_file_name = module_name
            test_func_name = func.__name__

        @ops_perftool(test_file_name, test_func_name, save_orig_trace=False)
        def wrapper(*args, **kwargs):
            logging.info(
                "\033[1;35m Current op and func: {}, {}. \033[0m".format(
                    type(args[0]).__name__, func.__name__
                )
            )
            st_time = time.time()
            func(*args, **kwargs)
            logging.info(
                "\033[1;30m Test time: %0.3f s. \033[0m" % (time.time() - st_time)
            )

        return wrapper

    return loader


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def get_comparison_dtype(a, b):
    # TODO: update this when promote_types supports bfloat16 and/or
    # isclose supports bfloat16.
    a_dtype = torch.float32 if a.dtype is torch.bfloat16 else a.dtype
    b_dtype = torch.float32 if b.dtype is torch.bfloat16 else b.dtype

    compare_dtype = torch.promote_types(a_dtype, b_dtype)

    # non-CUDA (CPU, for example) float16 -> float32
    # TODO: update this when isclose is implemented for CPU float16
    if compare_dtype is torch.float16 and (
        a.device != b.device or a.device.type != "cuda" or b.device.type != "cuda"
    ):
        compare_dtype = torch.float32

    return compare_dtype


def _has_sufficient_memory(device, size):
    if torch.device(device).type == "mlu":
        if not torch.mlu.is_available():
            return False
        gc.collect()
        torch.mlu.empty_cache()
        return (
            torch.mlu.get_device_properties(device).total_memory
            - torch.mlu.memory_allocated(device)
            >= size
        )

    if device == "xla":
        raise unittest.SkipTest("TODO: Memory availability checks for XLA?")

    if device != "cpu":
        raise unittest.SkipTest("Unknown device type")

    # CPU
    if not HAS_PSUTIL:
        raise unittest.SkipTest("Need psutil to determine if memory is sufficient")

    effective_size = size
    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size


def largeTensorTest(size, device=None):
    """Skip test if the device has insufficient memory to run the test

    size may be a number of bytes, a string of the form "N GB", or a callable

    If the test is a device generic test, available memory on the primary device will be checked.
    It can also be overriden by the optional `device=` argument.
    In other tests, the `device=` argument needs to be specified.
    """
    if isinstance(size, str):
        assert size.endswith("GB") or size.endswith("gb"), "only GB supported"
        size = 1024**3 * int(size[:-2])

    def inner(fn):
        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            size_bytes = size(self, *args, **kwargs) if callable(size) else size
            _device = (
                device
                if device is not None
                else "mlu:{0}".format(torch.mlu.current_device())
            )
            # large tensor test not support in 300 series
            if "MLU370" in torch.mlu.get_device_name():
                device_meg = (
                    f"large tensor test not support in {torch.mlu.get_device_name()}"
                )
                raise unittest.SkipTest(device_meg)
            if not _has_sufficient_memory(_device, size_bytes):
                raise unittest.SkipTest("Insufficient {} memory".format(_device))

            result = fn(self, *args, **kwargs)
            print(
                f"max memory usage {torch.mlu.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0}GB"
            )
            return result

        return dep_fn

    return inner


def mlufusion_on_and_off(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        orig_flg = torch.backends.mlufusion.enabled
        # concat the operator for lstm
        torch.backends.mlufusion.set_flags(False)
        func(*args, **kwargs)
        # use the fused cnnl operator for lstm
        torch.backends.mlufusion.set_flags(True)
        func(*args, **kwargs)
        torch.backends.mlufusion.set_flags(orig_flg)

    return wrapper


class TestCase(BaseTestCase):  # pylint: disable = R0904
    def to_non_dense(self, data, dim=None, distance=2):
        if not type(data) == torch.Tensor:
            print(
                "[Warning]: It's not available to convert an unknown object to non-dense type"
            )
            return data
        # convert the last channel as default.
        convert_dim = data.dim()
        if dim is not None:
            convert_dim = dim
        if convert_dim > data.dim():
            print(
                f"[Warning]: The max available expand dim for a {data.dim()} Tensor"
                f" is {data.dim()}, but got specified dim as {dim}."
            )
            convert_dim = data.dim()
        a = data.unsqueeze(convert_dim)
        b = torch.cat([a for _ in range(distance)], convert_dim)
        return b.select(dim=convert_dim, index=0)

    def to_mlu(self, input):
        """
        convert cpu-tensor into mlu-tensor based on self.data_type
        help to test both float32 and float16 data-type
        """
        if type(input) == torch.Tensor:
            if "PYTORCH_DATA_TYPE" in os.environ:
                mlu_input = input.type(self.data_type).mlu()
            else:
                mlu_input = input.mlu()
        else:
            mlu_input = input
        return mlu_input

    def to_mlu_dtype(self, input, data_type):
        if torch.is_tensor(input):
            mlu_input = input.to(data_type).mlu()
        else:
            mlu_input = input
        return mlu_input

    def to_device(self, input):
        if type(input) == torch.Tensor:
            mlu_input = input.mlu()
        else:
            mlu_input = input
        return mlu_input

    def convert_to_channel_last(self, x):
        if not isinstance(x, torch.Tensor):
            return x
        if x.dim() == 4:
            mlu_input = x.to(memory_format=torch.channels_last)
        elif x.dim() == 5:
            mlu_input = x.to(memory_format=torch.channels_last_3d)
        else:
            mlu_input = x
        return mlu_input

    def convert_to_is_non_overlapping_and_dense(self, x: torch.Tensor, seed: int):
        if not isinstance(x, torch.Tensor):
            return x
        nDim = x.dim()
        permute = [i for i in range(nDim)]
        random.seed(seed)
        random.shuffle(permute)

        # skip CL contiguous.
        def swap_list(lst, i, j):
            lst[i], lst[j] = lst[j], lst[i]
            return lst

        if nDim == 4 and permute == [0, 2, 3, 1]:
            permute = swap_list(permute, 1, 2)
            permute = swap_list(permute, 0, 3)
        if nDim == 5 and permute == [0, 2, 3, 4, 1]:
            permute = swap_list(permute, 1, 3)
            permute = swap_list(permute, 0, 4)
        new_size = [x.size()[i] for i in permute]
        new_stride = [x.stride()[i] for i in permute]
        for i in range(nDim):
            if new_size[i] == 1:
                new_stride[i] = random.randint(0, new_stride[i] + 1)
        result = x.as_strided(new_size, new_stride)
        return result

    def get_not_contiguous_tensor(self, x):
        self.assertTrue(isinstance(x, torch.Tensor), "Only support pytorch tensor.")
        dims = list(range(x.dim()))
        random.shuffle(dims)
        return x.permute(dims)

    def get_not_contiguous_tensor_container(self, x):
        self.assertTrue(isinstance(x, (list, tuple)), "Only support list or tuple.")
        result = []
        for item in x:
            out = self.get_not_contiguous_tensor(item)
            result.append(out)
        if isinstance(x, tuple):
            return tuple(result)
        else:
            return result

    def set_params(self):
        print("\n")
        # base config
        self.tensor_generator = torch.randn
        # set data type
        if "PYTORCH_DATA_TYPE" in os.environ:
            data_type = os.environ["PYTORCH_DATA_TYPE"].lower()
            if data_type == "half":
                self.data_type = torch.HalfTensor
            elif data_type == "float":
                self.data_type = torch.FloatTensor
            else:
                logging.error("Unknown data type!")
                exit(0)
        else:
            self.data_type = torch.FloatTensor

        return

    def setUp(self):
        # will be run before test
        self.set_params()

    def assertTensorsEqual(
        self,
        a,
        b,
        prec=None,
        message="",
        allow_inf=False,
        use_MSE=False,
        use_RAE=False,
        use_RMA=False,
    ):
        """unittest.TestCase"""
        if a.dtype == torch.bool:
            a = a.float()
        if b.dtype == torch.bool:
            b = b.float()
        epsilon = 1.0 / 16384
        self.assertEqual(a.size(), b.size(), message)
        if a.numel() > 0:
            a = self.optional_fake_half_cpu_inputs(a)
            b = self.optional_fake_half_cpu_inputs(b)
            # check that NaNs are in the same locations
            nan_mask = a != a
            self.assertTrue(torch.equal(nan_mask, b != b), message)
            diff = a - b
            diff[nan_mask] = 0
            a = a.clone()
            b = b.clone()
            a[nan_mask] = 0
            b[nan_mask] = 0
            # inf check if allow_inf=True
            if allow_inf:
                inf_mask = (a == float("inf")) | (a == float("-inf"))
                self.assertTrue(
                    torch.equal(inf_mask, (b == float("inf")) | (b == float("-inf"))),
                    message,
                )
                diff[inf_mask] = 0
                a[inf_mask] = 0
                b[inf_mask] = 0
            # TODO: implement abs on CharTensor
            if diff.is_signed() and "CharTensor" not in diff.type():
                diff = diff.abs()
            if use_MSE:
                diff = diff.abs().pow(2).sum()
                a_pow_sum = a.pow(2).sum()
                if diff <= (2 * epsilon) * (2 * epsilon):
                    diff = 0.0
                if a_pow_sum <= epsilon:
                    a_pow_sum = a_pow_sum + epsilon
                diff = torch.div(diff, (a_pow_sum * 1.0))
                self.assertLessEqual(diff.sqrt(), prec, message)
            elif use_RAE:
                diff = diff.abs().sum()
                a_sum = a.abs().sum()
                if a_sum == 0:
                    self.assertEqual(a, b, message)
                else:
                    diff = torch.div(diff, a_sum)
                    self.assertLessEqual(diff, prec, message)
            elif use_RMA:
                a_mean = a.abs().mean()
                b_mean = b.abs().mean()
                if a_mean == 0:
                    self.assertEqual(a, b, message)
                else:
                    diff = torch.div((a_mean - b_mean).abs(), a_mean)
                    self.assertLessEqual(diff, prec, message)
            else:
                max_err = diff.max()
                self.assertLessEqual(max_err, prec, message)

    dtype_precisions = {
        torch.float16: (0.001, 1e-5),
        torch.bfloat16: (0.016, 1e-5),
        torch.float32: (1.3e-6, 1e-5),  # currently is fp32 onchip
        torch.float64: (1.3e-6, 1e-5),
        torch.complex32: (0.001, 1e-5),
        torch.complex64: (1.3e-6, 1e-5),
        torch.complex128: (1.3e-6, 1e-5),  # currently is complex64 onchip
    }

    def TensorGenerator(self, shape, dtype, func=lambda x: x):
        if dtype.is_floating_point:
            cpu_tensor = torch.randn(shape).to(torch.half).to(torch.float)
            mlu_tensor = func(cpu_tensor.to("mlu").to(dtype))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor
        elif dtype.is_complex:
            cpu_tensor = torch.randn(shape, dtype=dtype)
            mlu_tensor = func(cpu_tensor.to("mlu"))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor
        elif dtype == torch.bool:
            cpu_tensor = torch.randint(0, 2, shape, dtype=dtype)
            mlu_tensor = func(cpu_tensor.to("mlu"))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor
        else:
            cpu_tensor = torch.randint(100, shape, dtype=dtype)
            mlu_tensor = func(cpu_tensor.to("mlu"))
            cpu_tensor = func(cpu_tensor)
            return cpu_tensor, mlu_tensor

    def _getDefaultRtolAndAtol(self, dtype0, dtype1):
        rtol = max(
            self.dtype_precisions.get(dtype0, (0, 0))[0],
            self.dtype_precisions.get(dtype1, (0, 0))[0],
        )
        atol = max(
            self.dtype_precisions.get(dtype0, (0, 0))[1],
            self.dtype_precisions.get(dtype1, (0, 0))[1],
        )

        return rtol, atol

    def optional_fake_half_cpu_inputs(self, tensor):
        if self.data_type == torch.HalfTensor:
            if isinstance(tensor, tuple):
                tensor = tuple(
                    x.type(torch.HalfTensor).type(torch.FloatTensor) for x in tensor
                )
            else:
                tensor = tensor.type(torch.HalfTensor).type(torch.FloatTensor)
                tensor[tensor == float("inf")] = 65504
            return tensor
        else:
            return tensor

    def assertNotWarnRegex(self, callable, regex=""):
        pattern = re.compile(regex)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            callable()
            self.assertFalse(any([re.match(pattern, str(w.message)) for w in ws]))

    if sys.version_info < (3, 2):
        # assertRegexpMatches renamed to assertRegex in 3.2
        assertRegex = unittest.TestCase.assertRegexpMatches
        # assertRaisesRegexp renamed to assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    if sys.version_info < (3, 5):
        # assertNotRegexpMatches renamed to assertNotRegex in 3.5
        assertNotRegex = unittest.TestCase.assertNotRegexpMatches


class OutputRedirector(object):
    """
    Class used to redirect standard output or another stream.
    """

    escape_char = "\b"

    def __init__(self, stream, threaded=False):
        self.current_stream = stream
        self.current_streamfd = self.current_stream.fileno()
        self.output_text = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start redirecting the stream data.
        """
        self.output_text = ""
        # Save a copy of the stream:
        self.original_streamfd = os.dup(self.current_streamfd)  # pylint: disable= W0201
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.current_streamfd)
        os.close(self.pipe_in)

    def stop(self):
        """
        Stop redirecting the stream data and save the text in `output_text`.
        """
        # Print the escape character to make the readOutput method stop:
        self.current_stream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.current_stream.flush()
        self.readOutput()
        # Close the pipe:
        os.close(self.pipe_out)
        # Restore the original stream:
        os.close(self.current_streamfd)
        os.dup2(self.original_streamfd, self.current_streamfd)
        # Close the duplicate stream:
        os.close(self.original_streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `output_text`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.current_stream.encoding)
            if not char or self.escape_char in char:
                break
            self.output_text += char


# cases which cause core dump and randomly error
known_error_lst = [
    "test_where_scalar_valid_combination_mlu_int64",
    "test_where_scalar_valid_combination_mlu_int16",
    "test_where_scalar_valid_combination_mlu_int8",
    "test_where_scalar_valid_combination_mlu_uint8",
]


def runAllTests(cls):
    """
    Execute all test cases in test_torch.py , output reports and
    execute the regression tests
    """
    name_lst = ["has_error", "op_name", "info"]
    out_frame = pd.DataFrame(columns=name_lst)
    passed = 0
    total = 0
    dataframe_lst = []

    for name in dir(cls):
        has_error = "no"
        if name[:4] == "test" and name not in known_error_lst:
            total += 1
            std_out = OutputRedirector(sys.stdout)
            std_err = OutputRedirector(sys.stderr)
            std_out.start()
            std_err.start()
            try:
                getattr(cls, name)()
            except Exception as e:  # pylint: disable = W0703
                info = sys.exc_info()
                print(info[0], ":", info[1])
                has_error = "yes"

            std_out.stop()
            std_err.stop()
            out = std_out.output_text + std_err.output_text
            if out != "":
                has_error = "yes"
            else:
                passed += 1
            row = pd.DataFrame(
                {name_lst[0]: [has_error], name_lst[1]: [name], name_lst[2]: [out]}
            )
            dataframe_lst.append(row)

    out_frame = pd.concat(dataframe_lst, ignore_index=True)
    print(
        "class ",
        cls.__class__.__name__,
        " passed ",
        passed,
        " cases. ",
        "ratio is ",
        passed / total,
    )
    out_frame.to_csv("./" + cls.__class__.__name__ + ".csv")

    # Check the regression test
    err_str = ""
    reg_csv = pd.read_csv("./Reg" + cls.__class__.__name__ + ".csv")
    for i in range(reg_csv.shape[0]):
        reg_test = reg_csv.iloc[i]
        out_test = out_frame.iloc[i]
        if reg_test["has_error"] == "no" and out_test["has_error"] == "yes":
            err_str += reg_test["op_name"]
            err_str += ", "
            print("ERROR OP: ", out_test["op_name"])
            print("ERROR INFO: ", out_test["info"])
    if err_str != "":
        # raise Exception("Regression test failed with these ops: ", err_str)
        warnings.warn("Regression test failed with these ops: " + err_str)


def runFnLst(cls, lst):
    """
    Test specified cases
    """
    for name in lst:
        getattr(cls, name)()


@contextmanager
def disable_functorch():
    guard = torch._C._DisableFuncTorch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


@contextlib.contextmanager
def freeze_rng_state():
    # no_dispatch needed for test_composite_compliance
    # Some OpInfos use freeze_rng_state for rng determinism, but
    # test_composite_compliance overrides dispatch for all torch functions
    # which we need to disable to get and set rng state
    with no_dispatch(), disable_functorch():
        rng_state = torch.get_rng_state()
        if torch.mlu.is_available():
            mlu_rng_state = torch.mlu.get_rng_state()
    try:
        yield
    finally:
        # Modes are not happy with torch.mlu.set_rng_state
        # because it clones the state (which could produce a Tensor Subclass)
        # and then grabs the new tensor's data pointer in generator.set_state.
        #
        # In the long run torch.mlu.set_rng_state should probably be
        # an operator.
        with no_dispatch(), disable_functorch():
            if torch.mlu.is_available():
                torch.mlu.set_rng_state(mlu_rng_state)
            torch.set_rng_state(rng_state)


# capture stdout/stderr from c++ and python
# The source is https://stackoverflow.com/a/29834357
class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """

    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char
