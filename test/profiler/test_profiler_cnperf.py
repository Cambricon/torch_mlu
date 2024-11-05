import sys
import os
import unittest
import logging
import subprocess

import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)

command_record = (
    os.getenv("NEUWARE_HOME")
    + "/bin/cnperf-cli record --capture_range=cnProfilerApi python "
    + cur_dir
    + "/test_cnperf.py"
)

command_report = (
    os.getenv("NEUWARE_HOME") + "/bin/cnperf-cli report 2>&1 |tee cnperf_report"
)

command_clean = "rm -rf dltrace_data && rm cnperf_report"


def check_string_in_file(file_path, search_string):
    try:
        with open(file_path, "r") as file:
            # Read the file content
            content = file.read()

            # Check if the search string is in the content
            if search_string in content:
                print(f"'{search_string}' is found in the file.")
                return True
            else:
                print(f"'{search_string}' is NOT found in the file.")
                return False
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return False


class TestProfilerCnperf(TestCase):
    # @unittest.skip("not test")
    def test_profiler_cnperf(self):
        subprocess.run(
            command_record, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        subprocess.run(
            command_report, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.assertTrue(check_string_in_file("./cnperf_report", "cnProfilerStart"))
        self.assertTrue(check_string_in_file("./cnperf_report", "cnProfilerStop"))
        self.assertTrue(check_string_in_file("./cnperf_report", "cnnlMatMulEx"))
        self.assertTrue(check_string_in_file("./cnperf_report", "1 [total]"))
        self.assertFalse(check_string_in_file("./cnperf_report", "MLUConvCo1"))
        subprocess.run(
            command_clean, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


if __name__ == "__main__":
    unittest.main()
