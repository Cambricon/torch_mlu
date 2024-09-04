import sys
import os
import re
import unittest
from torch_mlu.utils.collect_env import get_pretty_env_info

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import TestCase


class TestCollectEnv(TestCase):
    def test_smoke(self):
        info_output = get_pretty_env_info()
        self.assertTrue(info_output.count("\n") >= 34)

    def test_version_format(self):
        info_output = get_pretty_env_info()

        mlu_driver_version = re.search(
            r"MLU driver version: v(\d+\.\d+\.\d+)", info_output
        )
        self.assertIsNotNone(
            mlu_driver_version, "Incorrect {MLU driver version} format."
        )

        cnrt_runtime_version = re.search(
            r"cnrt runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        cndev_runtime_version = re.search(
            r"cndev runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        cndrv_runtime_version = re.search(
            r"cndrv runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        cnpapi_runtime_version = re.search(
            r"cnpapi runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        self.assertIsNotNone(
            cnrt_runtime_version, "Incorrect {cnrt runtime version} format."
        )
        self.assertIsNotNone(
            cndev_runtime_version, "Incorrect {cndev runtime version} format."
        )
        self.assertIsNotNone(
            cndrv_runtime_version, "Incorrect {cndrv runtime version} format."
        )
        self.assertIsNotNone(
            cnpapi_runtime_version, "Incorrect {cnpapi runtime version} format."
        )

        cndali_runtime_version = re.search(
            r"cndali runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        self.assertIsNotNone(
            cndali_runtime_version, "Incorrect {cndali runtime version} format."
        )

        cnnl_runtime_verson = re.search(
            r"cnnl runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        mluops_runtime_version = re.search(
            r"mlu-ops runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        cncl_runtime_version = re.search(
            r"cncl runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        cncv_runtime_version = re.search(
            r"cncv runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        cnnlextra_runtime_version = re.search(
            r"cnnl-extra runtime version: (v\d+\.\d+\.\d+|N/A)", info_output
        )
        self.assertIsNotNone(
            cnnl_runtime_verson, "Incorrect {cnnl runtime version} format."
        )
        self.assertIsNotNone(
            mluops_runtime_version, "Incorrect {mlu-ops runtime version} format."
        )
        self.assertIsNotNone(
            cncl_runtime_version, "Incorrect {cncl runtime version} format."
        )
        self.assertIsNotNone(
            cncv_runtime_version, "Incorrect {cncv runtime version} format."
        )
        self.assertIsNotNone(
            cnnlextra_runtime_version, "Incorrect {cnnl-extra runtime version} format."
        )


if __name__ == "__main__":
    unittest.main()
