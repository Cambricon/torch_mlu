"""
    * Testcase used to create .xml of printed logs for failed testcases that should
    * have generated .xml log after execution but didn't due to errors such
    * as coredump, etc.
    * Env variables:
    * FAILED_TEST_MODULE - the failing test_module
    * FAILED_LOG_FILE - the file containing printed logs of the failed test_module
    *
    * This case is only used with pytest.
"""
import os
import pytest

FAIL_OP_LIST = []
LOG_FILE = ""

if os.environ.get("FAILED_TEST_MODULE"):
    FAIL_OP_LIST = [os.environ["FAILED_TEST_MODULE"]]
if os.environ.get("FAILED_LOG_FILE"):
    LOG_FILE = os.environ["FAILED_LOG_FILE"]


@pytest.mark.parametrize("test_module", FAIL_OP_LIST)
def test_op_fail(test_module):
    if not os.path.exists(LOG_FILE):
        assert False, f"{test_module} failed but no log found at {LOG_FILE}"
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        assert False, f'{test_module} failed without .xml: \n {"".join(f.readlines())}'
