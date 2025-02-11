import os
import sys
import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import shell


class TestDistributedWrapper:
    def test_distributed_wrapper(self):
        run_cmd = [sys.executable, "test_distributed.py"]
        test_directory = os.path.dirname(os.path.abspath(__file__))
        return_code = shell(run_cmd, cwd=test_directory)
        assert return_code == 0, "test_distributed.py failed!"

        # test avoid record stream
        run_cmd = [sys.executable, "test_distributed.py", "--avoid_record_streams", "1"]
        return_code = shell(run_cmd, cwd=test_directory)
        assert return_code == 0, "test_distributed.py avoid record streams failed!"


if __name__ == "__main__":
    unittest.main()
